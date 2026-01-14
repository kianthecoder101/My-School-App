import streamlit as st
import io
import json
import base64
import urllib.parse
from PIL import Image
import numpy as np

# --- Google Sheets (gspread) connection (lazy) ---
def get_gsheets_connection():
    """
    Connect to Google Sheets using a service account stored in st.secrets['gcp_service_account'].
    Accepts dict, JSON string, or base64-encoded JSON string.
    Returns a gspread client or None on failure.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.warning(f"Missing gspread/google-auth packages: {e}")
        return None

    info = st.secrets.get("gcp_service_account")
    if not info:
        st.warning("No service account info found in st.secrets['gcp_service_account'].")
        return None

    service_info = None
    if isinstance(info, dict):
        service_info = info
    else:
        if isinstance(info, str):
            try:
                service_info = json.loads(info)
            except Exception:
                try:
                    decoded = base64.b64decode(info)
                    service_info = json.loads(decoded)
                except Exception as e:
                    st.warning(f"Failed to parse gcp_service_account secret: {e}")
                    return None
        else:
            st.warning("Unrecognized gcp_service_account secret format.")
            return None

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(service_info, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.warning(f"Could not create gspread client: {e}")
        return None

# --- OCR (lazy) ---
def do_ocr(uploaded_file):
    """Lazy-import easyocr and run OCR on the uploaded image. Returns dict with 'result' or 'error'."""
    try:
        import easyocr
    except Exception as e:
        return {"error": f"Failed to import OCR libraries: {e}. Consider running OCR locally or using a different OCR backend."}

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

# --- Camera helpers (lazy imports) ---
def fetch_snapshot_http(url, timeout=10):
    """Fetch a single image via HTTP(S) (snapshot URL). Returns bytes or raises."""
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"Missing requests package: {e}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def fetch_frame_from_rtsp(stream_url, timeout_sec=5):
    """
    Capture a single frame from a stream URL using OpenCV.
    Returns JPEG bytes or raises an exception.
    """
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"Missing OpenCV (cv2) package: {e}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Unable to open stream URL. Check URL, credentials, and network.")

    # Give the stream a moment to buffer
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

    # Convert BGR to RGB, encode as JPEG bytes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, encoded = cv2.imencode(".jpg", frame_rgb)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return encoded.tobytes()

# --- UI ---
def phone_camera_section(app_url):
    st.header("Phone camera (mobile)")
    st.write("Open this app on your phone, then use the camera input below to take a photo from your phone.")
    # QR code via Google Chart API (no extra dependency)
    qr_url = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(qr_url, width=200, caption="Scan to open app on phone")
        if st.button("Copy link"):
            st.write(app_url)  # small fallback - user can copy manually
    with col2:
        st.markdown("Or use the built-in camera input (works in mobile browsers):")
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

def ip_camera_section():
    st.header("IP / Outdoor camera (RTSP / Snapshot URL)")
    st.write("If your camera provides an RTSP stream (rtsp://...) or a single-image snapshot URL (http://.../snapshot.jpg), enter it here.")
    stream_url = st.text_input("Stream / snapshot URL (can include credentials in URL)")
    st.write("Examples: rtsp://user:pass@192.168.1.100:554/stream or http://camera-ip/snapshot.jpg")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch snapshot (HTTP)"):
            if not stream_url:
                st.error("Enter a snapshot URL first.")
            else:
                try:
                    img_bytes = fetch_snapshot_http(stream_url)
                    st.image(img_bytes, caption="Snapshot (HTTP)", use_column_width=True)
                except Exception as e:
                    st.error(f"HTTP snapshot failed: {e}")
    with col2:
        if st.button("Capture frame from stream (RTSP/HTTP)"):
            if not stream_url:
                st.error("Enter a stream URL first.")
            else:
                try:
                    img_bytes = fetch_frame_from_rtsp(stream_url)
                    st.image(img_bytes, caption="Stream frame", use_column_width=True)
                except Exception as e:
                    st.error(f"Stream capture failed: {e}")
    st.markdown(
        "If your camera cannot be reached directly from this host (e.g., it's behind NAT), consider configuring the camera to push snapshots to a public HTTP endpoint, upload to S3, or use an intermediate device that forwards frames."
    )

def main():
    st.title("My School App — Camera + OCR + Sheets")
    st.markdown("Use your phone camera or connect an outdoor camera via stream URL or snapshot URL.")

    # App URL for QR code — try to guess from request headers/environment where possible; otherwise ask the user.
    default_app_url = st.secrets.get("app_url") if st.secrets.get("app_url") else ""
    app_url = st.text_input("App public URL (used for QR code / phone open):", value=default_app_url)
    if not app_url:
        st.info("Paste your app's public URL here to enable QR code linking to phone.")

    # Phone camera UI
    phone_camera_section(app_url or "https://your-app-url.example")

    st.divider()

    # IP camera UI
    ip_camera_section()

    st.divider()

    # Google Sheets connection (kept available)
    client = get_gsheets_connection()
    if client:
        st.success("GSheets connection ready.")
        st.info("Use client.open_by_key(SHEET_ID) in the console or expand app to read/write sheets.")
    else:
        st.info("No GSheets connection available (working offline).")

    st.divider()

    # OCR upload fallback
    st.header("Upload image for OCR")
    uploaded = st.file_uploader("Upload an image (if you prefer)", type=["png", "jpg", "jpeg"])
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

if __name__ == "__main__":
    main()
