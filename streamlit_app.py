import streamlit as st
import io
import urllib.parse
import re
import time
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

# -------------------------
# 1. OCR Logic (Lightweight Tesseract)
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
        # --psm 7 tells Tesseract to look for a single line of text (a plate)
        text = pytesseract.image_to_string(image_np, config='--psm 7')
        return text.strip()
    except Exception:
        return ""

# -------------------------
# Config / Helpers
# -------------------------
PLATE_LIST_PATH = Path("list.txt")
REANNOUNCE_COOLDOWN_MIN = 5  

def now_utc():
    return datetime.now(timezone.utc)

def normalize_plate(s: str) -> str:
    if not s: return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def load_plate_list():
    if PLATE_LIST_PATH.exists():
        try:
            df = pd.read_csv(PLATE_LIST_PATH)
            return {normalize_plate(str(row['PlateNumber'])): row['StudentName'] 
                    for _, row in df.iterrows()}
        except Exception as e:
            st.warning(f"Failed to read list.txt: {e}")
            return {}
    return {}

# -------------------------
# Session State
# -------------------------
def ensure_state():
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []
    if "plate_list" not in st.session_state:
        st.session_state["plate_list"] = load_plate_list()
    if "last_seen" not in st.session_state:
        st.session_state["last_seen"] = {}

# -------------------------
# WebRTC Factory
# -------------------------
def webrtc_transformer_factory(sensitivity):
    try:
        from streamlit_webrtc import VideoProcessorBase
        import av
        import cv2
    except Exception:
        return None

    class PlateProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_check = 0

        def video_frame_callback(self, frame):
            img = frame.to_ndarray(format="bgr24")
            curr_time = time.time()

            # Process every 2 seconds to keep CPU low
            if curr_time - self.last_check > 2.0:
                self.last_check = curr_time
                try:
                    # Apply Sensitivity (Contrast Enhancement)
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    enhancer = ImageEnhance.Contrast(pil_img)
                    pil_img = enhancer.enhance(sensitivity)
                    
                    # Convert to grayscale for Tesseract
                    processed_img = np.array(pil_img.convert('L'))
                    
                    text = perform_ocr(processed_img)
                    norm = normalize_plate(text)
                    plate_map = st.session_state.get("plate_list", {})
                    
                    if norm in plate_map:
                        now = now_utc()
                        last = st.session_state["last_seen"].get(norm)
                        
                        if not last or (now - last).total_seconds() > (REANNOUNCE_COOLDOWN_MIN * 60):
                            st.session_state["last_seen"][norm] = now
                            student = plate_map[norm]
                            entry = {
                                "name": student, 
                                "plate": norm, 
                                "time": now.strftime("%H:%M:%S")
                            }
                            st.session_state["detected_log"].insert(0, entry)
                except Exception:
                    pass

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return PlateProcessor

# -------------------------
# Main UI
# -------------------------
def main():
    st.set_page_config(page_title="School Pickup Scanner", layout="wide")
    ensure_state()

    st.sidebar.title("Scanner Settings")
    # Sensitivity helps in low light or blurry conditions
    sens = st.sidebar.slider("OCR Sensitivity (Contrast)", 1.0, 3.0, 1.5, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"Database: {len(st.session_state['plate_list'])} Students")
    if st.sidebar.button("Reload list.txt"):
        st.session_state["plate_list"] = load_plate_list()
        st.rerun()

    st.title("🚗 School Pickup Scanner")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            
            webrtc_streamer(
                key=f"scanner-v1-{sens}", # Key reset on slider change
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: webrtc_transformer_factory(sens)(),
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False
                },
            )
        except Exception as e:
            st.error(f"Hardware Error: {e}")
        
        st.info("Ensure you are on HTTPS. If the camera doesn't turn on, check 'Manage App' logs for memory errors.")

    with col2:
        st.subheader("Pickup Queue")
        if st.session_state["detected_log"]:
            for item in st.session_state["detected_log"][:10]:
                st.success(f"🎓 **{item['name']}**\nPlate: {item['plate']} | {item['time']}")
        else:
            st.info("No plates detected yet.")

        if st.button("Clear Queue"):
            st.session_state["detected_log"] = []
            st.rerun()

if __name__ == "__main__":
    main()
