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

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

twilio_sid = os.getenv("TWILIO_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN") 

print("Welcome To Practice Page")

def draw_landmarks(image_pil, landmarks):
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
    for x, y in points:
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))

    return image_pil

def cam():

    webrtc_ctx = webrtc_streamer(
        key="camera",
        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
                    {"urls": ["stun:stun1.l.google.com:19302"]},  # Google STUN server (สำรอง)
                    {
                        "urls": ["turn:global.turn.twilio.com"],
                        "username": twilio_sid,
                        "credential": twilio_auth_token
            }
                ]
            }
        ),
        media_stream_constraints={"video": True, "audio": False},  
    )

def main():
    image_path = "Images/downdog.png"
    image_pil = Image.open(image_path)
    image_np = np.array(image_pil)
    landmarks = get_landmarks(image_np)
    annotated_image = draw_landmarks(image_pil, landmarks)
    graph = get_graph(landmarks)
    predicted_class = predict_pose(graph)
    predicted_key_area = predict_key_area(graph)


    # Front-end
    st.title("Yoga")

    cols = st.columns(2)

    with cols[0]:
        st.image(
            annotated_image, 
            caption = predicted_class, 
            use_container_width=True
        )
        st.markdown(predicted_key_area)

    with cols[1]:
        cam()

if __name__ == "__main__":
    main()

# peerConnection.oniceconnectionstatechange = function(event) {
#     console.log("ICE Connection State Change: " + peerConnection.iceConnectionState);
#     if (peerConnection.iceConnectionState === "disconnected") {
#         console.log("ICE connection has been disconnected");
#     }
# };

# peerConnection.onicecandidate = function(event) {
#     if (event.candidate) {
#         console.log("New ICE candidate: " + event.candidate.candidate);
#     }
# };



