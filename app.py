import streamlit as st
import torch
from PIL import Image
import numpy as np
import io

# Load local YOLOv7 model
model_path = 'best.pt'
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=model_path, force_reload=True)

# Streamlit app
st.title("Fine-tuned YOLOv7")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Display loading spinner
    with st.spinner('Please wait...'):
        # Perform inference
        results = model(image)

        # Save the image with detections
        results_img = results.render()[0]  # results.render() returns list of images
        results_img = Image.fromarray(results_img)

        st.image(results_img, caption='Processed Image.', use_column_width=True)
        st.success('Done!')

# Run the Streamlit app
# Command to run: streamlit run your_script_name.py
