"""Interactive Streamlit interface for the edge detection tool."""

import cv2
import numpy as np
import streamlit as st

from edge_detection import detect_edges


st.set_page_config(page_title="Edge Detection Tool", layout="wide")
st.title("Edge Detection Tool")
st.write("Compare Canny, Sobel, and Laplacian boundaries on your own image.")
uploaded = st.file_uploader("Upload a PNG or JPEG", type=["png", "jpg", "jpeg"])
method = st.selectbox("Method", ["canny", "sobel", "laplacian"], format_func=str.title)
blur = st.select_slider("Gaussian blur kernel", options=[0, 3, 5, 7, 9, 11], value=3,
                        help="Larger values suppress noise but can erase fine detail.")
kernel = st.select_slider("Detector kernel", options=[1, 3, 5, 7], value=3,
                          help="Canny uses 3 when 1 is selected.")
if method == "canny":
    low, high = st.slider("Canny thresholds (low, high)", 0, 255, (50, 150))
    if low >= high:
        st.warning("Set the upper threshold above the lower threshold.")
        st.stop()
else:
    low, high = 50, 150

if uploaded is None:
    st.info("Upload an image to see its edge map.")
    st.stop()
image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
if image is None or image.dtype != np.uint8:
    st.error("Please upload a valid 8-bit PNG or JPEG.")
    st.stop()
try:
    result = detect_edges(image, method, low=low, high=high, kernel=kernel, blur=blur)
except (ValueError, cv2.error) as exc:
    st.error(f"Could not process image: {exc}")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Original")
    if image.ndim == 2:
        st.image(image, clamp=True, use_container_width=True)
    else:
        code = cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
        st.image(cv2.cvtColor(image, code), clamp=True, use_container_width=True)
with right:
    st.subheader(f"{method.title()} edges")
    st.image(result, clamp=True, use_container_width=True)

ok, encoded = cv2.imencode(".png", result)
if ok:
    st.download_button("Download edge map (PNG)", encoded.tobytes(),
                       file_name=f"{method}_edges.png", mime="image/png")
st.caption(f"Image: {image.shape[1]} × {image.shape[0]} pixels. Output: 8-bit grayscale.")
