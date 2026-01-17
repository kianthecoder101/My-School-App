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
# 1. OCR Logic (Optimized for Low RAM)
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
        # PSM 7 is the lightest/fastest mode for single lines
        return pytesseract.image_to_string(image_np, config='--psm 7').strip()
    except:
        return ""

# -------------------------
# Config & State
# -------------------------
PLATE_LIST_PATH = Path("list.txt")

def ensure_state():
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []
    if "plate_list" not in st.session_state:
        if PLATE_LIST_PATH.exists():
            df = pd.read_csv(PLATE_LIST_PATH)
            st.session_state["plate_list"] = {
                re.sub(r"[^A-Z0-9]", "", str(row['PlateNumber']).upper()): row['StudentName'] 
                for _, row in df.iterrows()
            }
        else:
            st.session_state["plate_list"] = {}
    if "last_seen" not in st.session_state:
        st.session_state["last_seen"] = {}

# -------------------------
# WebRTC Factory (Lean Version)
# -------------------------
def webrtc_transformer_factory(sensitivity):
    try:
        from streamlit_webrtc import VideoProcessorBase
        import av
        import cv2
    except:
        return None

    class PlateProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_check = 0

        def video_frame_callback(self, frame):
            # 1. Get frame
            img = frame.to_ndarray(format="bgr24")
            curr_time = time.time()

            # 2. Check every 3 seconds (Reduces CPU spikes)
            if curr_time - self.last_check > 3.0:
                self.last_check = curr_time
                try:
                    # 3. Convert to small grayscale (Uses 4x less RAM)
                    small_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    small_gray = cv2.resize(small_gray, (480, 360))
                    
                    # 4. Enhance
                    pil_img = Image.fromarray(small_gray)
                    enhancer = ImageEnhance.Contrast(pil_img)
                    processed_img = np.array(enhancer.enhance(sensitivity))
                    
                    # 5. OCR & Match
                    text = perform_ocr(processed_img)
                    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                    
                    plate_map = st.session_state.get("plate_list", {})
                    if norm in plate_map:
                        now = datetime.now(timezone.utc)
                        last = st.session_state["last_seen"].get(norm)
                        # 5 minute cooldown
                        if not last or (now - last).total_seconds() > 300:
                            st.session_state["last_seen"][norm] = now
                            entry = {"name": plate_map[norm], "plate": norm, "time": now.strftime("%H:%M")}
                            st.session_state["detected_log"].insert(0, entry)
                except:
                    pass

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return PlateProcessor

# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="Pickup App", layout="centered")
    ensure_state()

    st.title("🚗 Pickup Scanner")
    sens = st.sidebar.slider("Sensitivity", 1.0, 3.0, 1.5)

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        webrtc_streamer(
            key=f"cam-{sens}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: webrtc_transformer_factory(sens)(),
            async_processing=True,
            # Force lower resolution at the hardware level
            media_stream_constraints={
                "video": {"width": 480, "height": 360, "frameRate": 15},
                "audio": False
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    except Exception as e:
        st.error(f"Cam error: {e}")

    st.subheader("Queue")
    for item in st.session_state["detected_log"][:5]:
        st.success(f"{item['name']} ({item['plate']})")

if __name__ == "__main__":
    main()
