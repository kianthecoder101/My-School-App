import streamlit as st
import io

from PIL import Image
import numpy as np


def get_gsheets_connection():
    """Create a gsheets connection at runtime. Wrapped in try/except so failures are shown in-app rather than at import time."""
    try:
        return st.connection("gsheets", type="gsheets")
    except Exception as e:
        st.warning(f"Could not create gsheets connection: {e}")
        return None


def do_ocr(uploaded_file):
    """Lazy-import easyocr and run OCR on the uploaded image. This avoids importing torch/easyocr at module import time.
    uploaded_file is a Streamlit UploadedFile or a bytes-like object.
    Returns a dict with either 'result' or 'error'.
    """
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

        # Create reader (cpu-only)
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

    conn = get_gsheets_connection()
    if conn:
        st.success("GSheets connection created.")
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

                        # Optionally show raw result
                        with st.expander("Show raw OCR output"):
                            st.json(result)


if __name__ == "__main__":
    main()
