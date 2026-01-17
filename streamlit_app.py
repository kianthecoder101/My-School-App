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
# 1. OCR Logic (Defined First)
# -------------------------
def perform_ocr(image_np):
    try:
        import pytesseract
        # PSM 7 is for single lines; helps prevent RAM spikes
        return pytesseract.image_to_string(image_np, config='--psm 7').strip()
    except Exception as e:
        return ""

# -------------------------
# 2. State Management (Defined Second)
# -------------------------
def ensure_state():
    if "detected_log" not in st.session_state:
        st.session_state["detected_log"] = []
    
    if "plate_list" not in st.session_state:
        plate_file = Path("list.txt")
        if plate_file.exists():
            try:
                df = pd.read_csv(plate_file)
                # Normalize the plates in the list
                st.session_state["plate_list"] = {
                    re.sub(r"[^A-Z0-9]", "", str(row['PlateNumber']).upper()): row['StudentName'] 
                    for _, row in df.iterrows()
                }
            except:
                st.session_state["plate_list"] = {}
        else:
            st.session_state["plate_list"] = {}

# -------------------------
# 3. Main UI Function
# -------------------------
def main():
    st.set_page_config(page_title="Pickup Pro", layout="centered")
    
    # Run state check immediately
    ensure_state()

    st.title("🚗 School Pickup Scanner")
    st.write("Point camera at plate and click scan.")

    # SIDE-BY-SIDE BUTTONS
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        scan_now = st.button("📸 SCAN PLATE NOW", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state["detected_log"] = []
            st.rerun()

    # CAMERA FEED
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        
        ctx = webrtc_streamer(
            key="pickup-cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.error(f"Camera Component Error: {e}")
        return

    # SCAN LOGIC
    if scan_now:
        if ctx.video_receiver:
            try:
                img_frame = ctx.video_receiver.get_frame()
                if img_frame:
                    img = img_frame.to_ndarray(format="bgr24")
                    
                    with st.spinner("Analyzing..."):
                        import cv2
                        # Convert to grayscale for Tesseract
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        text = perform_ocr(gray)
                        norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                        
                        # Show what we scanned
                        st.image(gray, caption=f"Last Scan: {norm}", width=200)

                        plate_map = st.session_state.get("plate_list", {})
                        if norm in plate_map:
                            name = plate_map[norm]
                            st.session_state["detected_log"].insert(0, f"✅ {name} ({norm})")
                            st.toast(f"Matched {name}!", icon="🎓")
                        else:
                            st.warning(f"No match for: {norm}")
                else:
                    st.error("Camera is frozen or not sending frames.")
            except Exception as e:
                st.error(f"Scan failed: {e}")
        else:
            st.warning("Please click 'Start' on the camera first.")

    # LOG DISPLAY
    st.divider()
    st.subheader("Pickup Queue")
    if st.session_state["detected_log"]:
        for item in st.session_state["detected_log"][:10]:
            st.write(item)
    else:
        st.info("Log is empty.")

if __name__ == "__main__":
    main()
