import os
import streamlit as st
from scripts.predict_image import predict_image

st.set_page_config(page_title="Skin Lesion Classification", layout="centered")

st.title("Skin Lesion Classification with Explainability")
st.write("Upload a skin lesion image to classify it as benign or malignant.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    img_path = os.path.join("temp", uploaded_file.name)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, confidence, gradcam_path = predict_image(img_path)

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")

        st.subheader("Grad-CAM Explanation")
        st.image(gradcam_path, caption="Grad-CAM Heatmap", use_container_width=True)

        st.warning("For educational use only. Not for clinical diagnosis.")