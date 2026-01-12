import streamlit as st
import easyocr
import numpy as np
from PIL import Image

conn = st.connection("gsheets", type="gsheets")
df = conn.read()
st.dataframe(df)

# 1. SET THE PAGE DESIGN
st.set_page_config(page_title="Sabah School Scanner", page_icon="🏫")
st.title("📸 School Pickup Scanner")
st.write("Point camera at the car plate to find the student.")

# 2. CONNECT TO YOUR GOOGLE SHEET
# PASTE YOUR LINK BETWEEN THE QUOTES BELOW
URL = "https://docs.google.com/spreadsheets/d/1g-mSE1uNpLQihVmUEzWbTdEW1sVBULfYM4gQm2kRxZ8/edit?usp=sharing"

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet=URL)
except:
    st.error("Check your Google Sheet link!")

# 3. OPEN THE CAMERA
img_file = st.camera_input("Scan Plate")

if img_file:
    # Process the image
    image = Image.open(img_file)
    img_array = np.array(image)
    
    with st.spinner('Reading plate...'):
        # Initialize the AI Reader
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img_array)
        
        # Combine all letters the AI sees into one string
        full_text = "".join([res[1] for res in results]).upper().replace(" ", "")
        st.write(f"I saw: **{full_text}**")

        # 4. SEARCH THE DATABASE
        found = False
        for index, row in df.iterrows():
            clean_plate = str(row['PlateNumber']).upper().replace(" ", "")
            
            # If the plate from our sheet is inside the text the camera saw...
            if clean_plate in full_text and clean_plate != "":
                st.success(f"✅ STUDENT: {row['StudentName']}")
                st.info(f"📍 CLASS: {row['Class']}")
                st.balloons()
                found = True
                break
        
        if not found:
            st.warning("No match found. Try getting closer to the plate.")
