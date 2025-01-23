import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer
from skeleton import process_pose_image
from PIL import Image


st.set_page_config(
    page_title="My Multi-page App",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.title("Yoga")


# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


image_path = "Images/downdog.png"
image_pil = Image.open(image_path)

image_np = np.array(image_pil)

annotated_image = process_pose_image(image_np)

cols = st.columns(2)

with cols[0]:
    st.image(
        annotated_image, 
        caption="Annotated Image with Mediapipe Pose", 
        use_container_width=True
    )
with cols[1]:
    # webrtc_streamer(key="sample")
    webrtc_streamer(key="example", media_stream_constraints={"video": True,"audio": False })

# cols = st.columns(2)
# with cols[0]:
#     all_pose_btn = st.button("전체 자세", use_container_width=True)
#     if all_pose_btn:
#         st.switch_page("pages/all.py")
# with cols[1]:
#     custom_btn = st.button("맞춤 프로그램", icon="🧘", use_container_width=True)
#     if custom_btn:
#         st.switch_page("pages/custom.py")


# webrtc_streamer(key="example")


