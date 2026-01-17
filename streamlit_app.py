import streamlit as st
import re
import pandas as pd
from PIL import Image
from pathlib import Path

# 1. Setup Session State
if "detected_log" not in st.session_state:
    st.session_state["detected_log"] = []
if "plate_list" not in st.session_state:
    if Path("list.txt").exists():
        df = pd.read_csv("list.txt")
        st.session_state["plate_list"] = {
            re.sub(r"[^A-Z0-9]", "", str(row['PlateNumber']).upper()): row['StudentName'] 
            for _, row in df.iterrows()
        }
    else: st.session_state["plate_list"] = {}

# 2. UI Layout
st.title("🚗 Smooth Pickup Scanner")
st.write("Video should be smooth now. Click 'SCAN' when the car is in view.")

col1, col2 = st.columns([2, 1])

with col1:
    # Use WebRTC just for the visual feed (No AI processing in the loop)
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        
        ctx = webrtc_streamer(
            key="smooth-feed",
            mode=WebRtcMode.SENDRECV,
            # We remove the VideoProcessor entirely to keep it fast
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.error(f"Camera error: {e}")

    # 3. The "Manual Trigger" Button
    if ctx.video_receiver:
        if st.button("📸 SCAN PLATE NOW", use_container_width=True):
            img_frame = ctx.video_receiver.get_frame()
            if img_frame:
                img = img_frame.to_ndarray(format="bgr24")
                
                # Perform OCR only on command
                with st.spinner("Reading..."):
                    import pytesseract
                    import cv2
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config='--psm 7').strip()
                    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                    
                    if norm in st.session_state["plate_list"]:
                        name = st.session_state["plate_list"][norm]
                        st.session_state["detected_log"].insert(0, f"✅ {name} ({norm})")
                    else:
                        st.warning(f"Saw '{norm}' - No match found.")

with col2:
    st.subheader("Queue")
    for item in st.session_state["detected_log"][:10]:
        st.write(item)
