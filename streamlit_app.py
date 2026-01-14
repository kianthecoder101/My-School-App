import streamlit as st
import io
import json
import base64

from PIL import Image
import numpy as np


def get_gsheets_connection():
    """
    Connect to Google Sheets using a service account stored in st.secrets['gcp_service_account'].
    This function accepts three secret formats:
      - a mapping/dict (Streamlit Cloud can store JSON as a mapped object)
      - a JSON string
      - a base64-encoded JSON string

    Returns a gspread client or None on failure.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.warning(f"Missing gspread/google-auth packages: {e}")
        return None

    info = st.secrets.get("gcp_service_account")
    if not info:
        st.warning("No service account info found in st.secrets['gcp_service_account'].")
        return None

    # Normalize to a Python dict service_info
    service_info = None
    if isinstance(info, dict):
        service_info = info
    else:
        # info is likely a JSON string or base64-encoded JSON
        if isinstance(info, str):
            # try JSON
            try:
                service_info = json.loads(info)
            except Exception:
                # try base64 decode then JSON
                try:
                    decoded = base64.b64decode(info)
                    service_info = json.loads(decoded)
                except Exception as e:
                    st.warning(f"Failed to parse gcp_service_account secret: {e}")
                    return None
        else:
            st.warning("Unrecognized gcp_service_account secret format.")
            return None

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(service_info, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.warning(f"Could not create gspread client: {e}")
        return None


def do_ocr(uploaded_file):
    """Lazy-import easyocr and run OCR on the uploaded image. Returns dict with 'result' or 'error'."""
    try:
        import easyocr
    except Exception as e:
        return {"error": f"Failed to import OCR libraries: {e}. Consider running OCR locally or using a different OCR backend."}

    try:
        # Read image bytes and convert to numpy array
        if hasattr(uploaded_file, "read"):
            img_bytes = uploaded_file.read()
        else:
            img_bytes = uploaded_file
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(image)

        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(image_np)
        return {"result": result}
    except Exception as e:
        return {"error": f"OCR failed: {e}"}


def format_ocr_result(result):
    """Turn easyocr result into readable text and a small table structure."""
    lines = []
    for bbox, text, conf in result:
        lines.append({"text": text, "confidence": conf})
    return lines


def main():
    st.title("My School App")

    st.markdown(
        "This app connects to Google Sheets (when available) and can run OCR on uploaded images. "
        "OCR libraries are imported only when you click 'Run OCR' to avoid import-time failures."
    )

    client = get_gsheets_connection()
    if client:
        st.success("GSheets connection created.")
        st.info("You can open a sheet by ID with client.open_by_key(YOUR_SHEET_ID)")
    else:
        st.info("No GSheets connection available (continuing in offline mode).")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

        if st.button("Run OCR"):
            with st.spinner("Running OCR..."):
                ocr_out = do_ocr(uploaded)
                if "error" in ocr_out:
                    st.error(ocr_out["error"])
                else:
                    result = ocr_out["result"]
                    if not result:
                        st.info("No text detected.")
                    else:
                        formatted = format_ocr_result(result)
                        st.subheader("Detected text")
                        for i, row in enumerate(formatted, start=1):
                            st.write(f"{i}. {row['text']}  —  confidence: {row['confidence']:.2f}")

                        with st.expander("Show raw OCR output"):
                            st.json(result)


if __name__ == "__main__":
    main()
