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

# Lazy imports for heavy libs are done where needed.

# -------------------------
# Config / helpers
# -------------------------
PLATE_LIST_PATH = Path("list.txt")  # repo root file; one plate per line
REANNOUNCE_COOLDOWN_MIN = 5  # minutes before same plate is re-announced


def now_utc():
    return datetime.now(timezone.utc)


def normalize_plate(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def load_plate_list():
    if PLATE_LIST_PATH.exists():
        try:
            text = PLATE_LIST_PATH.read_text(encoding="utf-8")
            lines = [normalize_plate(l) for l in text.splitlines() if l.strip()]
            return set(lines)
        except Exception as e:
            st.warning(f"Failed to read {PLATE_LIST_PATH}: {e}")
            return set()
    else:
        return set()


# -------------------------
# Camera / stream helpers
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
# OCR helpers
# -------------------------
def ocr_on_image_bytes(img_bytes) -> list:
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
        texts = [t for (_b, t, _c) in out]
        return texts
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return []


# -------------------------
# TTS helper (gTTS)
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
# Session state helpers
# -------------------------
def ensure_state():
    if "monitoring" not in st.session_state:
        st.session_state["monitoring"] = False
    if "announced" not in st.session_state:
        st.session_state["announced"] = {}
    if "plate_list" not in st.session_state:
        st.session_state["plate_list"] = load_plate_list()
    if "last_frame" not in st.session_state:
        st.session_state["last_frame"] = None
    if "webrtc_announces" not in st.session_state:
        st.session_state["webrtc_announces"] = []


def update_plate_list_from_upload(uploaded_file):
    try:
        text = uploaded_file.getvalue().decode("utf-8")
        lines = [normalize_plate(l) for l in text.splitlines() if l.strip()]
        st.session_state["plate_list"] = set(lines)
        st.success(f"Loaded {len(lines)} entries from uploaded list.")
    except Exception as e:
        st.error(f"Failed to read uploaded list.txt: {e}")


def announce_plate(plate_norm: str, raw_text: str):
    now = now_utc()
    last = st.session_state["announced"].get(plate_norm)
    if last and (now - last) < timedelta(minutes=REANNOUNCE_COOLDOWN_MIN):
        return False, "recently_announced"
    text = f"Attention. Plate detected: {raw_text}."
    tts = generate_tts_mp3_bytes(text)
    if "error" in tts:
        return False, tts["error"]
    mp3 = tts["mp3"]
    st.session_state["announced"][plate_norm] = now
    st.success(f"Matched plate: {raw_text}")
    st.audio(mp3, format="audio/mp3")
    st.download_button(
        label=f"Download announcement for {raw_text}",
        data=mp3,
        file_name=f"announce-{plate_norm}-{int(now.timestamp())}.mp3",
        mime="audio/mpeg",
    )
    return True, "ok"


# -------------------------
# Option A: server-pull continuous monitoring
# -------------------------
def monitor_continuous(stream_url: str, interval_sec: float = 1.0):
    placeholder = st.empty()
    ensure_state()
    st.session_state["monitoring"] = True
    st.info("Live monitoring started — press STOP to end.")
    iteration = 0
    while st.session_state.get("monitoring", False):
        iteration += 1
        try:
            if stream_url.lower().startswith("rtsp://") or stream_url.lower().startswith("rtsps://"):
                img_bytes = fetch_frame_from_rtsp(stream_url, timeout_sec=3)
            else:
                img_bytes = fetch_snapshot_http(stream_url, timeout=5)
            placeholder.image(img_bytes, caption=f"Live frame #{iteration}", use_column_width=True)
            st.session_state["last_frame"] = img_bytes
            texts = ocr_on_image_bytes(img_bytes)
            for t in texts:
                norm = normalize_plate(t)
                if norm and norm in st.session_state["plate_list"]:
                    announce_plate(norm, t)
        except Exception as e:
            placeholder.error(f"Frame fetch / processing error: {e}")
        slept = 0.0
        step = 0.25
        while slept < interval_sec:
            time.sleep(step)
            slept += step
            if not st.session_state.get("monitoring", False):
                break
    placeholder.empty()
    st.success("Live monitoring stopped.")


# -------------------------
# Option B: browser push via streamlit-webrtc
# -------------------------
def webrtc_available():
    try:
        import streamlit_webrtc  # noqa: F401
        return True
    except Exception:
        return False


# Proper transformer subclassing VideoTransformerBase (imported lazily)
def webrtc_transformer_factory():
    try:
        from streamlit_webrtc import VideoTransformerBase  # type: ignore
        import av  # noqa: F401
    except Exception:
        return None

    class PlateTransformer(VideoTransformerBase):
        def __init__(self):
            self.reader = None
            self.last_announced = {}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")  # BGR
            # process frame (non-blocking best-effort)
            try:
                import easyocr
                if self.reader is None:
                    self.reader = easyocr.Reader(["en"], gpu=False)
                img_rgb = img[:, :, ::-1]
                out = self.reader.readtext(img_rgb)
                for (_bbox, text, _conf) in out:
                    norm = normalize_plate(text)
                    if norm and norm in st.session_state.get("plate_list", set()):
                        now_dt = now_utc()
                        last = self.last_announced.get(norm)
                        if not last or (now_dt - last).total_seconds() > REANNOUNCE_COOLDOWN_MIN * 60:
                            self.last_announced[norm] = now_dt
                            tts = generate_tts_mp3_bytes(f"Attention. Plate detected: {text}.")
                            if "mp3" in tts:
                                ev = {"plate": text, "mp3": tts["mp3"], "time": now_dt.isoformat()}
                                st.session_state.setdefault("webrtc_announces", []).append(ev)
            except Exception:
                # tolerate OCR failures; keep showing frames
                pass
            import av
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return PlateTransformer


# -------------------------
# UI: Monitor page with both sources + QR (webrtc for phone)
# -------------------------
def camera_monitor_section(app_url: str):
    st.header("Live monitoring (plates → announce)")
    st.markdown(
        "Use Browser (webrtc) from a phone for live push streaming, or use RTSP/HTTP stream for outdoor cameras (server-pull)."
    )
    ensure_state()

    # QR + webrtc instructions
    qr_col, info_col = st.columns([1, 2])
    with qr_col:
        st.subheader("Open on phone")
        if app_url:
            qr_url = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
            st.image(qr_url, width=200, caption="Scan to open app on phone")
            st.markdown(f"[Open on phone]({app_url})")
        else:
            st.info("Paste your public app URL into the sidebar to enable QR & link.")

    with info_col:
        st.subheader("Phone live (Browser → WebRTC) — recommended for mobile")
        st.write(
            "Open this page on your phone, allow camera, then select 'Browser (webrtc)' below and start camera. "
            "No RTSP exposure required."
        )

    st.markdown("---")

    st.subheader("Live source")
    src = st.radio("Choose live source", ["Browser (webrtc)", "RTSP / HTTP (server-pull)"])

    if src == "Browser (webrtc)":
        st.write("Browser (webrtc) selected. Open the app URL on your phone and allow camera access.")
        if not webrtc_available():
            st.error(
                "streamlit-webrtc is not installed. Add streamlit-webrtc, av, aiortc to requirements.txt and redeploy."
            )
        else:
            try:
                from streamlit_webrtc import webrtc_streamer  # noqa: E402
            except Exception as e:
                st.error(f"Failed to import streamlit-webrtc: {e}")
            else:
                Transformer = webrtc_transformer_factory()
                if Transformer is None:
                    st.error("Unable to create webrtc transformer (missing av/VideoTransformerBase).")
                else:
                    webrtc_ctx = webrtc_streamer(
                        key="webrtc",
                        mode="recvonly",
                        video_transformer_factory=Transformer,
                        media_stream_constraints={"video": True, "audio": False},
                        async_transform=True,
                    )
                    # show any TTS events produced by transformer
                    if st.session_state.get("webrtc_announces"):
                        for ev in st.session_state["webrtc_announces"]:
                            st.write(f"Detected plate: {ev['plate']} at {ev['time']}")
                            st.audio(ev["mp3"], format="audio/mp3")
                        st.session_state["webrtc_announces"].clear()

    else:
        st.write("Server-pull selected (RTSP/HTTP). Fill stream URL and start monitoring.")
        col1, col2 = st.columns([2, 1])
        with col1:
            stream_url = st.text_input("Camera stream or snapshot URL (rtsp:// or http:// ...):", key="stream_url")
            interval = st.number_input("Polling interval (sec)", min_value=0.5, max_value=30.0, value=1.0)
            max_iters = st.number_input("Max iterations (0 = infinite until STOP)", min_value=0, value=0)
            uploaded_list = st.file_uploader("Upload list.txt (optional) — one plate per line", type=["txt"])
            if uploaded_list:
                update_plate_list_from_upload(uploaded_list)
            st.write(f"Plates in list: {len(st.session_state['plate_list'])}")
        with col2:
            if not st.session_state["monitoring"]:
                if st.button("START monitoring"):
                    if not stream_url:
                        st.error("Enter a stream/snapshot URL first.")
                    else:
                        st.session_state["monitoring"] = True
                        monitor_continuous(stream_url, interval_sec=float(interval))
            else:
                if st.button("STOP monitoring"):
                    st.session_state["monitoring"] = False
                    st.experimental_rerun()
            if st.session_state.get("last_frame"):
                st.subheader("Last captured frame")
                st.image(st.session_state["last_frame"], use_column_width=True)
            st.subheader("Recently announced")
            recent = st.session_state["announced"]
            if recent:
                for plate, dt in sorted(recent.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{plate} — last announced {dt.isoformat()}")
            else:
                st.write("None yet")

    st.markdown("---")
    st.info("QR uses the Public app URL from the sidebar. Use ngrok for local HTTPS during testing.")


# -------------------------
# Announcements (session)
# -------------------------
def announcements_page():
    st.header("Announcements (session)")
    if "announcements" not in st.session_state:
        st.session_state["announcements"] = []
    with st.expander("Create an announcement (session only)"):
        t = st.text_input("Title")
        b = st.text_area("Body")
        if st.button("Add announcement"):
            if not t or not b:
                st.error("Title and body required")
            else:
                item = {"id": int(time.time() * 1000), "title": t, "body": b, "created_at": now_utc().isoformat()}
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


# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title="My School App", layout="wide")
    st.sidebar.title("My School App")
    page = st.sidebar.radio("Go to", ["Monitor (plates)", "Announcements", "Settings"])
    default_app_url = st.secrets.get("app_url", "")
    app_url = st.sidebar.text_input("Public app URL (for QR on phones)", value=default_app_url)
    if app_url:
        try:
            qr = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
            st.sidebar.image(qr, width=150, caption="Scan to open (if your URL is public HTTPS)")
            st.sidebar.markdown(f"[Open app on phone]({app_url})")
        except Exception:
            st.sidebar.write(f"[Open app on phone]({app_url})")
    else:
        st.sidebar.info("Add your public HTTPS app URL here to enable QR linking to phones (or use ngrok).")
    ensure_state()
    if page == "Monitor (plates)":
        camera_monitor_section(app_url or "")
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
        st.write(sorted(st.session_state.get("plate_list", [])))


if __name__ == "__main__":
    main()
