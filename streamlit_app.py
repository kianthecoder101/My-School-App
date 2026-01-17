import streamlit as st
import re
import time
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import threading

# -------------------------
# 1. OCR Logic (Runs in a separate thread)
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
        # Speed optimized config
        return pytesseract.image_to_string(image_np, config='--psm 7').strip()
    except:
        return ""

# -------------------------
# WebRTC Factory (Threaded)
# -------------------------
def webrtc_transformer_factory():
    try:
        from streamlit_webrtc import VideoProcessorBase
        import av
        import cv2
    except: return None

    class PlateProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_check = 0
            self.processing = False  # Track if a scan is already happening

        def video_frame_callback(self, frame):
            img = frame.to_ndarray(format="bgr24")
            curr_time = time.time()

            # Check every 3 seconds, but ONLY if not already busy
            if curr_time - self.last_check > 3.0 and not self.processing:
                self.last_check = curr_time
                # Launch OCR in a separate thread so the video doesn't freeze
                thread = threading.Thread(target=self._process_in_background, args=(img.copy(),))
                thread.start()

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def _process_in_background(self, img):
            self.processing = True
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text = perform_ocr(gray)
                norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                
                # Matching Logic
                if "plate_list" in st.session_state and norm in st.session_state["plate_list"]:
                    name = st.session_state["plate_list"][norm]
                    now = datetime.now(timezone.utc)
                    # Add to log if not seen recently
                    st.session_state["detected_log"].insert(0, f"{name} ({norm})")
            finally:
                self.processing = False

    return PlateProcessor

# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="Smooth Scanner")
    
    # Load List once
    if "plate_list" not in st.session_state:
        if Path("list.txt").exists():
            df = pd.read_csv("list.txt")
            st.session_state["plate_list"] = {
                re.sub(r"[^A-Z0-9]", "", str(row['PlateNumber']).upper()): row['StudentName'] 
                for _, row in df.iterrows()
            }
        else: st.session_state["plate_list"] = {}
    
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []

    st.title("🚗 Live Scanner")

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        webrtc_streamer(
            key="smooth-cam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=webrtc_transformer_factory,
            async_processing=True, # THIS keeps the video moving
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.error(f"Error: {e}")

    st.write("---")
    for item in st.session_state["detected_log"][:5]:
        st.success(item)

if __name__ == "__main__":
    main()
