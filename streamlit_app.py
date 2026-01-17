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
# OCR & Setup
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
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
# WebRTC Processor
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

            if curr_time - self.last_check > 3.0:
                self.last_check = curr_time
                try:
                    # Minimal processing to save RAM
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (480, 320))
                    pil_img = Image.fromarray(gray)
                    enhancer = ImageEnhance.Contrast(pil_img)
                    processed_img = np.array(enhancer.enhance(sensitivity))
                    
                    text = perform_ocr(processed_img)
                    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                    
                    plate_map = st.session_state.get("plate_list", {})
                    if norm in plate_map:
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

    st.title("🚗 Pickup Scanner")
    sens = st.sidebar.slider("Sensitivity", 1.0, 3.0, 1.5)

    # TWO WAYS TO SCAN: Live Video or Photo Capture
    tab1, tab2 = st.tabs(["Live Scanner", "Manual Photo (Fallback)"])

    with tab1:
        st.write("If this box disappears, use the 'Manual Photo' tab.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            webrtc_streamer(
                key=f"scanner-{sens}",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: webrtc_transformer_factory(sens)(),
                async_processing=True,
                # STRENGTHENED CONFIG: Adding Twilio or Google Relays
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                },
                media_stream_constraints={
                    "video": {"width": 480, "height": 320, "frameRate": 10},
                    "audio": False
                },
            )
        except Exception as e:
            st.error(f"Live Video Error: {e}")

    with tab2:
        st.write("Use this if your network blocks live streaming.")
        img_file = st.camera_input("Take a photo of the plate")
        if img_file:
            img = Image.open(img_file).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            processed = np.array(enhancer.enhance(sens))
            text = perform_ocr(processed)
            norm = re.sub(r"[^A-Z0-9]", "", text.upper())
            
            if norm in st.session_state["plate_list"]:
                name = st.session_state["plate_list"][norm]
                st.success(f"Matched: {name}")
                st.session_state["detected_log"].insert(0, {"name": name, "plate": norm, "time": "Now"})
            else:
                st.warning(f"Detected '{norm}' but no match in list.txt")

    st.divider()
    st.subheader("Pickup Queue")
    for item in st.session_state["detected_log"][:5]:
        st.success(f"**{item['name']}** - {item['plate']} ({item['time']})")

if __name__ == "__main__":
    main()
