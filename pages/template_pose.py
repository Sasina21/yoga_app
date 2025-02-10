import os
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from PIL import Image
import cv2
import json
import mediapipe as mp


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

print("Welcome To Template_pose Page")


def load_results():
    OUTPUT_FILE = "storage.json"

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    return {}

def cam():

    webrtc_ctx = webrtc_streamer(
        key="camera",
        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            }
        ),
        media_stream_constraints={"video": True, "audio": False},  
    )

def main():
    st.title("🧘‍♂️ Yoga Pose Assessment")

    query_params = st.query_params
    selected_pose = query_params.get("pose", None)

    if not selected_pose:
        st.error("No data. Please try again")
        return

    # โหลดข้อมูลภาพที่ตรงกับท่าโยคะที่เลือก
    results = load_results()
    filtered_images = [img for img, label in results.items() if label == selected_pose]

    if not filtered_images:
        st.error(f"No file path of {selected_pose}")
        return

    # เลือกรูปแรกที่เจอ
    image_path = filtered_images[0]  
    image_pil = Image.open(image_path)


    # UI
    st.title("Yoga")

    cols = st.columns(2)

    with cols[0]:
        st.image(
            image_pil, 
            caption=f"📸 Example Pose: {selected_pose}", 
            use_container_width=True
        )
        # st.markdown(predicted_key_area)

    with cols[1]:
        cam()

if __name__ == "__main__":
    main()



