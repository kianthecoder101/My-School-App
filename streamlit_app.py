import streamlit as st
import io
import json
import base64
import urllib.parse
from datetime import datetime, timezone
from PIL import Image
import numpy as np

# ---------- Utilities ----------
def now_utc():
    return datetime.now(timezone.utc)

def ensure_announcements_state():
    if "announcements" not in st.session_state:
        st.session_state["announcements"] = []

def make_card(title, subtitle, body, extra=None):
    st.markdown("**" + title + "**")
    if subtitle:
        st.caption(subtitle)
    st.write(body)
    if extra:
        st.write(extra)

# ---------- Google Sheets placeholder (kept for later) ----------
def get_gsheets_connection():
    """Placeholder for gspread connection; kept for later integration."""
    st.info("GSheets integration is currently paused. Add credentials and enable when ready.")
    return None

# ---------- OCR (lazy) ----------
def do_ocr(uploaded_file):
    """Lazy-import easyocr and run OCR on the uploaded image. Returns dict with 'result' or 'error'."""
    try:
        import easyocr
    except Exception as e:
        return {"error": f"Failed to import OCR libraries: {e}. Consider running OCR locally or using another OCR backend."}

    try:
        if hasattr(uploaded_file, "read"):
            img_bytes = uploaded_file.read()
        else:
            img_bytes = uploaded_file
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(image)

        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(image_np)
        return {"result": result}
    except Exception as e:
        return {"error": f"OCR failed: {e}"}

def format_ocr_result(result):
    lines = []
    for bbox, text, conf in result:
        lines.append({"text": text, "confidence": conf})
    return lines

# ---------- Camera helpers (lazy imports) ----------
def fetch_snapshot_http(url, timeout=10):
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"Missing requests package: {e}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def fetch_frame_from_rtsp(stream_url, timeout_sec=5):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"Missing OpenCV (cv2) package: {e}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Unable to open stream URL. Check URL, credentials, and network.")

    import time
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

# ---------- TTS helpers ----------
def generate_tts_mp3(text, lang="en"):
    """
    Generate mp3 bytes from text using gTTS (lazy import).
    Note: gTTS requires internet access at runtime.
    """
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

# ---------- Announcements management ----------
def add_announcement(title, body, author, publish_at=None):
    ensure_announcements_state()
    item = {
        "id": int(datetime.now().timestamp() * 1000),
        "title": title,
        "body": body,
        "author": author,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "publish_at": publish_at.isoformat() if publish_at else None,
    }
    st.session_state["announcements"].insert(0, item)  # newest first
    return item

def get_published_announcements(include_unpublished=False):
    ensure_announcements_state()
    out = []
    now = now_utc()
    for a in st.session_state["announcements"]:
        if a.get("publish_at"):
            try:
                pub = datetime.fromisoformat(a["publish_at"])
            except Exception:
                pub = None
        else:
            pub = None
        if include_unpublished or (pub is None or pub <= now):
            out.append(a)
    return out

# ---------- UI sections ----------
def home_section():
    st.header("Welcome to the automated student pickup dashboard")
    st.markdown(
        """
        This app combines camera capture, OCR, and announcements with TTS playback.
        Use the sidebar to navigate:
        - Camera: connect phone or IP cameras and run OCR
        - Announcements: create and play announcement audio (TTS)
        """
    )
    st.markdown("Built for quick testing — announcements persist in the session. Connect a real store later (Sheets, DB, S3).")

def camera_section(app_url):
    st.header("Camera & OCR")
    st.markdown("Use your phone camera, upload an image, or connect an IP camera stream/snapshot.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Phone")
        qr_url = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
        st.image(qr_url, width=180, caption="Scan to open on phone")
        photo = st.camera_input("Take a photo from your phone")
        if photo:
            st.image(photo, caption="Captured photo", use_column_width=True)
            if st.button("Run OCR on phone photo"):
                with st.spinner("Running OCR..."):
                    out = do_ocr(photo)
                    if "error" in out:
                        st.error(out["error"])
                    else:
                        formatted = format_ocr_result(out["result"])
                        st.write(formatted)

    with col2:
        st.subheader("IP / Outdoor camera")
        stream_url = st.text_input("Stream or snapshot URL (RTSP or HTTP JPG):")
        if st.button("Fetch snapshot (HTTP)"):
            if not stream_url:
                st.error("Enter a snapshot URL first.")
            else:
                try:
                    img_bytes = fetch_snapshot_http(stream_url)
                    st.image(img_bytes, caption="Snapshot (HTTP)", use_column_width=True)
                except Exception as e:
                    st.error(f"HTTP snapshot failed: {e}")

        if st.button("Capture frame from stream (RTSP/HTTP)"):
            if not stream_url:
                st.error("Enter a stream URL first.")
            else:
                try:
                    img_bytes = fetch_frame_from_rtsp(stream_url)
                    st.image(img_bytes, caption="Stream frame", use_column_width=True)
                except Exception as e:
                    st.error(f"Stream capture failed: {e}")

    st.divider()
    st.subheader("Upload image for OCR")
    uploaded = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)
        if st.button("Run OCR on uploaded image"):
            with st.spinner("Running OCR..."):
                ocr_out = do_ocr(uploaded)
                if "error" in ocr_out:
                    st.error(ocr_out["error"])
                else:
                    result = ocr_out["result"]
                    formatted = format_ocr_result(result)
                    st.write("Detected text:")
                    for i, row in enumerate(formatted, start=1):
                        st.write(f"{i}. {row['text']}  —  confidence: {row['confidence']:.2f}")

def announcements_section():
    st.header("Announcements")
    st.markdown("Create announcements and generate TTS audio for them. Audio playback requires internet (gTTS).")

    with st.expander("Create a new announcement", expanded=True):
        title = st.text_input("Title")
        body = st.text_area("Body", height=120)
        author = st.text_input("Author (optional)")
        publish = st.checkbox("Schedule publish time", value=False)
        publish_at = None
        if publish:
            dt = st.date_input("Publish date", value=datetime.now().date())
            tm = st.time_input("Publish time", value=datetime.now().time().replace(microsecond=0))
            publish_at = datetime.combine(dt, tm).astimezone(timezone.utc)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add announcement"):
                if not title or not body:
                    st.error("Title and body are required.")
                else:
                    item = add_announcement(title, body, author or "Anonymous", publish_at)
                    st.success("Announcement added.")
                    st.experimental_rerun()
        with col2:
            if st.button("Clear announcements (session only)"):
                st.session_state["announcements"] = []
                st.success("Cleared announcements for this session.")
                st.experimental_rerun()

    st.divider()
    show_all = st.checkbox("Show unpublished / future announcements", value=False)
    announcements = get_published_announcements(include_unpublished=show_all)
    if not announcements:
        st.info("No announcements yet.")
    for a in announcements:
        with st.container():
            st.subheader(a["title"])
            meta = f"By {a.get('author','Anonymous')} — created {a.get('created_at')}"
            if a.get("publish_at"):
                meta += f" — publishes {a.get('publish_at')}"
            st.caption(meta)
            st.write(a["body"])

            bcol1, bcol2 = st.columns([1, 3])
            with bcol1:
                if st.button(f"Play TTS ▶️ (#{a['id']})"):
                    with st.spinner("Generating audio..."):
                        tts = generate_tts_mp3(f"{a['title']}. {a['body']}")
                        if "error" in tts:
                            st.error(tts["error"])
                        else:
                            st.audio(tts["mp3"], format="audio/mp3")
                            st.download_button(
                                label="Download MP3",
                                data=tts["mp3"],
                                file_name=f"announcement-{a['id']}.mp3",
                                mime="audio/mpeg",
                            )
            with bcol2:
                st.write("")  # spacing; could add more actions here

    st.divider()
    if st.button("Download all announcements (JSON)"):
        ensure_announcements_state()
        data = json.dumps(st.session_state["announcements"], indent=2)
        st.download_button("Download JSON", data=data, file_name="announcements.json", mime="application/json")

# ---------- Main app ----------
def main():
    st.set_page_config(page_title="My School App", layout="wide")
    st.sidebar.title("My School App")
    nav = st.sidebar.radio("Navigate", ["Home", "Camera", "Announcements"])

    default_app_url = st.secrets.get("app_url") if st.secrets.get("app_url") else ""
    app_url = st.sidebar.text_input("Public app URL (for phone QR)", value=default_app_url)

    if nav == "Home":
        home_section()
        st.sidebar.markdown("---")
        st.sidebar.info("Tip: Use the Camera page for phone and IP camera captures. Create announcements on the Announcements page.")
    elif nav == "Camera":
        camera_section(app_url or "https://your-app-url.example")
    elif nav == "Announcements":
        announcements_section()

if __name__ == "__main__":
    main()
