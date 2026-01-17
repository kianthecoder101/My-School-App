import streamlit as st
import re
import pandas as pd
from pathlib import Path

# ... (Keep your OCR and Session State logic at the top) ...

def main():
    st.set_page_config(page_title="Pickup Pro", layout="centered")
    ensure_state()

    st.title("🚗 School Pickup Scanner")

    # --- BUTTON ROW ---
    # Creating columns to put the Scan button right next to where the camera controls are
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        # This button will sit on the left
        scan_now = st.button("📸 CLICK TO SCAN PLATE", use_container_width=True)

    with col_btn2:
        # This acts as a reminder or a "Clear" button to sit next to it
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state["detected_log"] = []
            st.rerun()

    # --- CAMERA SECTION ---
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    
    ctx = webrtc_streamer(
        key="integrated-feed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # --- SCAN LOGIC ---
    if scan_now:
        if ctx.video_receiver:
            try:
                img_frame = ctx.video_receiver.get_frame()
                if img_frame:
                    img = img_frame.to_ndarray(format="bgr24")
                    
                    with st.spinner("Processing..."):
                        import pytesseract
                        import cv2
                        # Optimized Grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        text = pytesseract.image_to_string(gray, config='--psm 7').strip()
                        norm = re.sub(r"[^A-Z0-9]", "", text.upper())
                        
                        # Show a small preview of the scan
                        st.image(gray, caption=f"Last Read: {norm}", width=150)

                        if norm in st.session_state["plate_list"]:
                            name = st.session_state["plate_list"][norm]
                            st.session_state["detected_log"].insert(0, f"✅ {name} ({norm})")
                            st.balloons() # Visual celebration for a match!
                        else:
                            st.error(f"No match for: {norm}")
            except Exception as e:
                st.error(f"Scan failed: {e}")
        else:
            st.warning("Please click 'Start' on the video feed first!")

    st.divider()
    st.subheader("Pickup Queue")
    for item in st.session_state["detected_log"][:10]:
        st.write(item)

if __name__ == "__main__":
    main()
