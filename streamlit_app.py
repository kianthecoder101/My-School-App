import streamlit as st
import io
import re
import time
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

# -------------------------
# OCR Logic (Hardened)
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
        # PSM 7 + OSD 0 is the lightest configuration possible
        return pytesseract.image_to_string(image_np, config='--psm 7').strip()
    except:
        return ""

PLATE_LIST_PATH = Path("list.txt")

def ensure_state():
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []
    if "plate_list" not in st.session_state:
        if PLATE_LIST_PATH.exists():
            try:
                df = pd.read_csv(PLATE_LIST_PATH)
                st.session_state["plate_list"] = {
                    re.sub(r"[^A-Z0-9]", "", str(row['PlateNumber']).upper()): row['StudentName'] 
                    for _, row in df.iterrows()
                }
            except: st.session_state["plate_list"] = {}
        else:
            st.session_state["plate_list"] = {}
    if "last_seen" not in st.session_state:
        st.session_state["last_seen"] = {}

# -------------------------
# WebRTC Factory (Stability Focused)
# -------------------------
def webrtc_transformer_factory(sensitivity):
    try:
        from streamlit_webrtc import VideoProcessorBase
        import av
        import cv2
    except: return None

    class PlateProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_check = 0

        def video_frame_callback(self, frame):
            img = frame.to_ndarray(format="bgr24")
            curr_time = time.time()

            # Check every 4 seconds (Slower check = More stability)
            if curr_time - self.last_check > 4.0:
                self.last_check = curr_time
                try:
                    # Resize to tiny dimensions for OCR (Plate only needs small res)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray, (320, 240)) 
                    
                    text = perform_ocr(small)
                    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                    
                    plate_map = st.session_state.get("plate_list", {})
                    if norm and norm in plate_map:
                        now = datetime.now(timezone.utc)
                        last = st.session_state["last_seen"].get(norm)
                        if not last or (now - last).total_seconds() > 300:
                            st.session_state["last_seen"][norm] = now
                            entry = {"name": plate_map[norm], "plate": norm, "time": now.strftime("%H:%M")}
                            st.session_state["detected_log"].insert(0, entry)
                except: pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return PlateProcessor

# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="Pickup App", layout="centered")
    ensure_state()

    st.title("🚗 School Pickup")
    
    # 2. Add a Start Button to prevent auto-loading crash
    run = st.checkbox("Toggle Camera On/Off", value=False)

    if run:
        st.info("Loading camera... If it turns black, wait 5 seconds.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            webrtc_streamer(
                key="stable-cam",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: webrtc_transformer_factory(1.5)(),
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={
                    "video": {"width": 320, "height": 240, "frameRate": 10}, # Ultra low res
                    "audio": False
                },
            )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Camera is currently OFF. Check the box above to start.")

    st.divider()
    st.subheader("Queue")
    for item in st.session_state["detected_log"][:5]:
        st.success(f"**{item['name']}** - {item['plate']}")

if __name__ == "__main__":
    main()
