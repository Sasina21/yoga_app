import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
from data_preparation import get_landmarks

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

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

def main():
    run = st.button("카메라 활성화", type="primary")

    frame_window = st.image([])

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        # if results.pose_landmarks:
        #     mp.solutions.drawing_utils.draw_landmarks(
        #         frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        #     )

        #     for idx, landmark in enumerate(results.pose_landmarks.landmark):
        #         x = landmark.x  
        #         y = landmark.y  
        #         z = landmark.z  
        #         visibility = landmark.visibility  

        #         st.text(f"Landmark {idx}: x={x:.2f}, y={y:.2f}, z={z:.2f}, visibility={visibility:.2f}")

        frame_window.image(frame)

    cap.release()

image_path = "Images/downdog.png"
image_pil = Image.open(image_path)
image_np = np.array(image_pil)
landmarks = get_landmarks(image_np)
annotated_image = draw_landmarks(image_pil, landmarks)


# Front-end
st.title("Yoga")

cols = st.columns(2)

with cols[0]:
    st.image(
        annotated_image, 
        caption="Annotated Image with Mediapipe Pose", 
        use_container_width=True
    )

with cols[1]:
     main()

# if __name__ == "__main__":
#     main()


# rtc_configuration = RTCConfiguration({
#     "iceServers": [
#         {"urls": ["stun:stun.l.google.com:19302"]},  # STUN Server
#         # {
#         #     "urls": ["turn:172.17.0.2:3478?transport=tcp"],  # TURN Server
#         #     "username": "your-username",
#         #     "credential": "your-password"
#         # }
#     ]
# })

# with cols[1]:
#     webrtc_logger = logging.getLogger("streamlit_webrtc")
#     webrtc_logger.info("Starting WebRTC streamer")
#     webrtc_streamer(
#         key="example",
#         rtc_configuration=rtc_configuration,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True
        
#     )
#     logging.info("WebRTC application is running")


