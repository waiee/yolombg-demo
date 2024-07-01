import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
import os

# Set the path to the YOLOv7 repository
yolov7_repo_path = 'C:/Users/user/Downloads/projects/MBG-YOLOv7/yolov7'  # Update this with the path to your YOLOv7 repository
sys.path.append(yolov7_repo_path)

# Import YOLOv7 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box

# Load local YOLOv7 model
model_path = 'best.pt'  # Change this if your model is in a different path

# Load the model
device = torch.device('cpu')
model = attempt_load(model_path, map_location=device)
model.eval()

# Streamlit app
st.title("Fine-tuned YOLOv7")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Display loading spinner
    with st.spinner('Wait for it...'):
        # Convert image to tensor
        img = np.array(image)
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)

        # Process results
        for det in pred:  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.size).round()

                # Draw boxes and labels on image
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=2)

        st.image(image, caption='Processed Image.', use_column_width=True)
        st.success('Done!')
