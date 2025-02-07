import os
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
from data_preparation import get_landmarks, get_graph
from class_prediction import predict_pose
from key_area_prediction import predict_key_area

print("Welcome To Predict Page")

def draw_landmarks(image_pil, landmarks, predicted_key_area):
    if image_pil is None or landmarks is None:
        print("Error: Image or Landmarks is None")
        return None

    # image_pil = image_pil.convert("RGB")
    draw = ImageDraw.Draw(image_pil)

    width, height = image_pil.size
    print(f"width: {width}, height: {height}")
    mp_pose = mp.solutions.pose
    pose_connections = mp_pose.POSE_CONNECTIONS

    points = [(int(lm["x"] * width), int(lm["y"] * height)) for lm in landmarks]

    # edge
    for start_idx, end_idx in pose_connections:
        if start_idx < len(points) and end_idx < len(points):
            draw.line([points[start_idx], points[end_idx]], fill=(0, 0, 255), width=3)

    # landmarks
    for idx, (x, y) in enumerate(points):
        radius = 4
        if predicted_key_area[idx] == 1:
            # Blue color for key = 1
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
        else:
            # Gray color for key = 0
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 255))

    return image_pil

def main():
    # Use Streamlit to upload image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open image
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)
        landmarks = get_landmarks(image_np)
        graph = get_graph(landmarks)
        predicted_class = predict_pose(graph)
        predicted_key_area = predict_key_area(graph)
        annotated_image = draw_landmarks(image_pil, landmarks, predicted_key_area)

        # Display the annotated image and prediction
        st.title("Yoga Pose Prediction")
        st.image(
            annotated_image, 
            caption=predicted_class, 
            use_container_width=True
        )
        st.markdown(f"Predicted Key Areas: {predicted_key_area}")

if __name__ == "__main__":
    main()



