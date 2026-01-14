import streamlit as st
import io
import json
import base64
import urllib.parse
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image
import numpy as np

# -------------------------
# Config / helpers
# -------------------------
PLATE_LIST_PATH = Path("list.txt")  # file in repo root with one plate per line
REANNOUNCE_COOLDOWN_MIN = 5  # minutes before the same plate can be announced again

def now_utc():
    return datetime.now(timezone.utc)

def normalize_plate(s: str) -> str:
    """Normalize OCR text to an uppercase alphanumeric-only string for matching."""
    if not s:
        return ""
    return re.sub(r'[^A-Z0-9]', '', s.upper())

def load_plate_list():
    """Load plate whitelist from list.txt (one plate per line). Returns set of normalized plates."""
    if PLATE_LIST_PATH.exists():
        try:
            text = PLATE_LIST_PATH.read_text(encoding="utf-8")
            lines = [normalize_plate(l) for l in text.splitlines() if l.strip()]
            return set(lines)
        except Exception as e:
            st.warning(f"Failed to read {PLATE_LIST_PATH}: {e}")
            return set()
    else:
        # allow upload fallback
        return set()

# -------------------------
# Camera / stream helpers (lazy imports)
# -------------------------
def fetch_snapshot_http(url: str, timeout: int = 10) -> bytes:
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"Missing requests package: {e}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def fetch_frame_from_rtsp(stream_url: str, timeout_sec: int = 5) -> bytes:
    """
    Try to capture one frame from RTSP/stream using OpenCV and return JPEG bytes.
    """
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"Missing OpenCV (cv2) package: {e}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Unable to open stream URL. Check URL, credentials, and network.")

    start = time.time()
    frame = None
    while time.time() - start < timeout_sec:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.2)

    cap.release()
    if frame is None:
        raise RuntimeError("Failed to capture a frame from the stream.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, encoded = cv2.imencode(".jpg", frame_rgb)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return encoded.tobytes()

# -------------------------
# OCR helpers (lazy import)
# -------------------------
def ocr_on_image_bytes(img_bytes) -> list:
    """
    Run OCR on image bytes and return a list of recognized text strings (raw).
    Requires easyocr installed. Returns empty list on failure.
    """
    try:
        import easyocr
    except Exception as e:
        st.error(f"OCR unavailable (easyocr missing): {e}")
        return []

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(image)
        reader = easyocr.Reader(["en"], gpu=False)
        out = reader.readtext(image_np)
        # out items: (bbox, text, conf)
        texts = [t for (_b, t, _c) in out]
        return texts
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return []

# -------------------------
# TTS helper (gTTS lazy)
# -------------------------
def generate_tts_mp3_bytes(text: str, lang: str = "en"):
    try:
        from gtts import gTTS
    except Exception as e:
        return {"error": f"Missing gTTS package: {e}. Add 'gtts' to requirements."}
    try:
        tts = gTTS(text=text, lang=lang)
        bio = io.BytesIO()
        tts.write_to_fp(bio)
        bio.seek(0)
        return {"mp3": bio.read()}
    except Exception as e:
        return {"error": f"TTS generation failed: {e}"}

# -------------------------
# Monitoring logic
# -------------------------
def ensure_state():
    if "monitoring" not in st.session_state:
        st.session_state["monitoring"] = False
    if "announced" not in st.session_state:
        # map normalized_plate -> last_announced_utc (datetime)
        st.session_state["announced"] = {}
    if "plate_list" not in st.session_state:
        st.session_state["plate_list"] = load_plate_list()
    if "last_frame" not in st.session_state:
        st.session_state["last_frame"] = None

def update_plate_list_from_upload(uploaded_file):
    try:
        text = uploaded_file.getvalue().decode("utf-8")
        lines = [normalize_plate(l) for l in text.splitlines() if l.strip()]
        st.session_state["plate_list"] = set(lines)
        st.success(f"Loaded {len(lines)} entries from uploaded list.")
    except Exception as e:
        st.error(f"Failed to read uploaded list.txt: {e}")

def announce_plate(plate_norm: str, raw_text: str):
    """Create an announcement entry, TTS it and play it in the UI; avoid re-announcing within cooldown."""
    now = now_utc()
    last = st.session_state["announced"].get(plate_norm)
    if last and (now - last) < timedelta(minutes=REANNOUNCE_COOLDOWN_MIN):
        # skip reannounce within cooldown
        return False, "recently_announced"
    # prepare announcement text
    text = f"Attention. Plate detected: {raw_text}."
    # generate TTS
    tts = generate_tts_mp3_bytes(text)
    if "error" in tts:
        return False, tts["error"]
    mp3 = tts["mp3"]
    # record announce time
    st.session_state["announced"][plate_norm] = now
    # show audio player and download
    st.success(f"Matched plate: {raw_text}")
    st.audio(mp3, format="audio/mp3")
    st.download_button(
        label=f"Download announcement for {raw_text}",
        data=mp3,
        file_name=f"announce-{plate_norm}-{int(now.timestamp())}.mp3",
        mime="audio/mpeg",
    )
    return True, "ok"

def monitor_loop(stream_url: str, interval_sec: float = 2.0, max_iterations: int = 0):
    """
    Poll the stream (RTSP or HTTP snapshot) repeatedly while st.session_state.monitoring is True.
    interval_sec: seconds between frames
    max_iterations: non-zero to limit loop for testing
    """
    placeholder = st.empty()
    iteration = 0
    st.info("Monitoring started. Click STOP to end monitoring.")
    while st.session_state["monitoring"]:
        iteration += 1
        if max_iterations and iteration > max_iterations:
            break
        try:
            if stream_url.lower().startswith("rtsp://") or stream_url.lower().startswith("rtsps://"):
                img_bytes = fetch_frame_from_rtsp(stream_url)
            else:
                # try HTTP snapshot
                img_bytes = fetch_snapshot_http(stream_url)
            # display frame
            placeholder.image(img_bytes, caption=f"Live frame {iteration}", use_column_width=True)
            st.session_state["last_frame"] = img_bytes
            # run OCR and match
            texts = ocr_on_image_bytes(img_bytes)
            matched_any = False
            for t in texts:
                norm = normalize_plate(t)
                if not norm:
                    continue
                if norm in st.session_state["plate_list"]:
                    matched_any = True
                    ok, msg = announce_plate(norm, t)
                    if not ok and msg == "recently_announced":
                        st.info(f"{t} was announced recently; skipping reannounce.")
            if not texts:
                st.write("No text detected on this frame.")
        except Exception as e:
            st.error(f"Frame fetch / processing error: {e}")

        # small sleep but allow UI to update
        for _ in range(int(max(1, interval_sec * 2))):
            time.sleep(0.5)
            # if user clicked stop we want to break promptly
            if not st.session_state["monitoring"]:
                break

        # re-run to keep Streamlit responsive if monitoring still true
        if st.session_state["monitoring"]:
            st.experimental_rerun()
    placeholder.empty()
    st.info("Monitoring stopped.")

# -------------------------
# UI
# -------------------------
def camera_monitor_section():
    st.header("Live monitoring (plates -> announce)")

    st.markdown(
        "Enter your camera's stream URL (RTSP) or a snapshot URL (HTTP). The app will poll the feed, run OCR on frames, "
        "and, if a recognized plate matches an entry in list.txt, it will create and play an announcement (TTS)."
    )

    # ensure state
    ensure_state()

    col1, col2 = st.columns([2, 1])
    with col1:
        stream_url = st.text_input("Camera stream or snapshot URL (rtsp:// or http:// ...):", key="stream_url")
        interval = st.number_input("Polling interval seconds", min_value=1.0, max_value=30.0, value=2.0)
        max_iters = st.number_input("Max iterations (0 = infinite until STOP)", min_value=0, value=0)
        st.write("Note: host must be able to reach the camera. Cameras behind NAT need forwarding or a push endpoint.")
        uploaded_list = st.file_uploader("Upload list.txt (optional) — one plate per line", type=["txt"])
        if uploaded_list:
            update_plate_list_from_upload(uploaded_list)

        # show loaded plates count
        st.write(f"Plates in list: {len(st.session_state['plate_list'])}")

    with col2:
        st.subheader("Controls")
        if not st.session_state["monitoring"]:
            if st.button("START monitoring"):
                if not stream_url:
                    st.error("Enter a stream/snapshot URL first.")
                else:
                    st.session_state["monitoring"] = True
                    # start monitoring (runs in the same request; we use experimental_rerun in loop)
                    monitor_loop(stream_url, interval_sec=interval, max_iterations=int(max_iters))
        else:
            if st.button("STOP monitoring"):
                st.session_state["monitoring"] = False
                st.experimental_rerun()

        if st.session_state.get("last_frame"):
            st.subheader("Last captured frame")
            st.image(st.session_state["last_frame"], use_column_width=True)

        # show announced history
        st.subheader("Recently announced")
        recent = st.session_state["announced"]
        if recent:
            for plate, dt in sorted(recent.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{plate} — last announced {dt.isoformat()}")
        else:
            st.write("None yet")

    st.markdown("---")
    st.info(
        "QR code: ensure your Public app URL (in the sidebar) is a public HTTPS URL. If QR wasn't working, try adding the URL manually "
        "to your phone browser. You can also use an ngrok HTTPS URL for local testing."
    )

def announcements_page():
    st.header("Announcements (session)")
    # small reuse of previous announcement UI
    if "announcements" not in st.session_state:
        st.session_state["announcements"] = []
    with st.expander("Create an announcement (session only)"):
        t = st.text_input("Title")
        b = st.text_area("Body")
        if st.button("Add announcement"):
            if not t or not b:
                st.error("Title and body required")
            else:
                item = {
                    "id": int(time.time() * 1000),
                    "title": t,
                    "body": b,
                    "created_at": now_utc().isoformat(),
                }
                st.session_state["announcements"].insert(0, item)
                st.success("Added")
    if st.session_state["announcements"]:
        for a in st.session_state["announcements"]:
            st.subheader(a["title"])
            st.write(a["body"])
            if st.button(f"Play TTS (#{a['id']})"):
                tts = generate_tts_mp3_bytes(f"{a['title']}. {a['body']}")
                if "error" in tts:
                    st.error(tts["error"])
                else:
                    st.audio(tts["mp3"], format="audio/mp3")
    else:
        st.info("No announcements yet.")

def main():
    st.set_page_config(page_title="My School App", layout="wide")
    st.sidebar.title("My School App")
    page = st.sidebar.radio("Go to", ["Monitor (plates)", "Announcements", "Settings"])
    # app_url usage for QR generation; user should put their public app URL here
    default_app_url = st.secrets.get("app_url", "")
    app_url = st.sidebar.text_input("Public app URL (for QR on phones)", value=default_app_url)

    # show quick link / QR helper
    if app_url:
        try:
            qr = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
            st.sidebar.image(qr, width=150, caption="Scan to open (if your URL is public HTTPS)")
        except Exception:
            st.sidebar.write(f"[Open app on phone]({app_url})")
    else:
        st.sidebar.info("Add your public HTTPS app URL here to enable QR linking to phones (or use ngrok for local tests).")

    if page == "Monitor (plates)":
        camera_monitor_section()
    elif page == "Announcements":
        announcements_page()
    else:
        st.header("Settings")
        st.markdown(
            "- Place a file named list.txt in the repository root (one plate per line). "
            "You can also upload list.txt in the Monitor page to load it for the session.\n"
            f"- Current list.txt present: {'yes' if PLATE_LIST_PATH.exists() else 'no'}"
        )
        st.write("Loaded plates (session):")
        ensure_state()
        st.write(sorted(st.session_state.get("plate_list", [])))

if __name__ == "__main__":
    main()
