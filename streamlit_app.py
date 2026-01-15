import streamlit as st
import io
import urllib.parse
import re
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image
import numpy as np

# -------------------------
# Config / helpers
# -------------------------
PLATE_LIST_PATH = Path("list.txt")  # Your CSV file
REANNOUNCE_COOLDOWN_MIN = 5  

def now_utc():
    return datetime.now(timezone.utc)

def normalize_plate(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def load_plate_list():
    if PLATE_LIST_PATH.exists():
        try:
            # Read as CSV and clean up plate numbers
            df = pd.read_csv(PLATE_LIST_PATH)
            # Create a dictionary for quick lookup: {PLATE: STUDENT_NAME}
            mapping = {normalize_plate(str(row['PlateNumber'])): row['StudentName'] 
                       for _, row in df.iterrows()}
            return mapping
        except Exception as e:
            st.warning(f"Failed to read {PLATE_LIST_PATH}: {e}")
            return {}
    return {}

# -------------------------
# OCR helper
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
        return [t for (_b, t, _c) in out]
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return []

# -------------------------
# Session state
# -------------------------
def ensure_state():
    if "monitoring" not in st.session_state:
        st.session_state["monitoring"] = False
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []
    if "plate_list" not in st.session_state:
        st.session_state["plate_list"] = load_plate_list()
    if "last_seen" not in st.session_state:
        st.session_state["last_seen"] = {}

# -------------------------
# WebRTC Factory
# -------------------------
def webrtc_transformer_factory():
    try:
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode
        import av
        import easyocr
    except Exception:
        return None

    class PlateProcessor(VideoProcessorBase):
        def __init__(self):
            self.reader = easyocr.Reader(["en"], gpu=False)
            self.last_check = 0

        def video_frame_callback(self, frame):
            img = frame.to_ndarray(format="bgr24")
            curr_time = time.time()

            # Process OCR every 2 seconds to save CPU/Battery
            if curr_time - self.last_check > 2.0:
                self.last_check = curr_time
                img_rgb = img[:, :, ::-1]
                out = self.reader.readtext(img_rgb)
                
                for (_bbox, text, _conf) in out:
                    norm = normalize_plate(text)
                    plate_map = st.session_state.get("plate_list", {})
                    
                    if norm in plate_map:
                        now = now_utc()
                        last = st.session_state["last_seen"].get(norm)
                        
                        if not last or (now - last).total_seconds() > (REANNOUNCE_COOLDOWN_MIN * 60):
                            st.session_state["last_seen"][norm] = now
                            student = plate_map[norm]
                            entry = {"name": student, "plate": norm, "time": now.strftime("%H:%M:%S")}
                            st.session_state["detected_log"].insert(0, entry)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return PlateProcessor

# -------------------------
# UI Sections
# -------------------------
def camera_monitor_section(app_url: str):
    st.header("Live Monitoring (Silent)")
    ensure_state()

    # QR Code for Phone access
    if app_url:
        qr_url = "https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=" + urllib.parse.quote(app_url)
        st.image(qr_url, width=150, caption="Scan to open on phone")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Scanner")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            Processor = webrtc_transformer_factory()
            
            if Processor:
                webrtc_streamer(
                    key="plate-scanner",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=Processor,
                    async_processing=True,
                    # FIX: STUN server for connection timeout issues
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={
                        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                        "audio": False
                    },
                )
        except Exception as e:
            st.error(f"WebRTC Error: {e}")

    with col2:
        st.subheader("Identified Students")
        if st.session_state["detected_log"]:
            for item in st.session_state["detected_log"][:15]:
                st.success(f"**{item['name']}**\n{item['plate']} at {item['time']}")
        else:
            st.info("No students detected yet.")
            
        if st.button("Clear History"):
            st.session_state["detected_log"] = []
            st.rerun()

def main():
    st.set_page_config(page_title="School App", layout="wide")
    st.sidebar.title("School Plate Scanner")
    
    page = st.sidebar.radio("Navigation", ["Monitor", "Student Database"])
    app_url = st.sidebar.text_input("Public App URL (for QR)", value="")
    
    ensure_state()

    if page == "Monitor":
        camera_monitor_section(app_url)
    else:
        st.header("Student Database")
        st.write(f"Total students loaded: {len(st.session_state['plate_list'])}")
        
        # Display the loaded CSV data
        if st.session_state['plate_list']:
            df_display = pd.DataFrame(
                [(k, v) for k, v in st.session_state['plate_list'].items()],
                columns=["Plate Number", "Student Name"]
            )
            st.table(df_display)

        if st.button("Reload list.txt"):
            st.session_state["plate_list"] = load_plate_list()
            st.success("Database updated!")

if __name__ == "__main__":
    main()
