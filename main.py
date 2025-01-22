import streamlit as st
from streamlit_webrtc import webrtc_streamer
import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.title("Yoga")

cols = st.columns(2)
with cols[0]:
    all_pose_btn = st.button("전체 자세", use_container_width=True)
with cols[1]:
    custom_btn = st.button("맞춤 프로그램", icon="🧘", use_container_width=True)


# webrtc_streamer(key="example")


