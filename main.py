import streamlit as st
import numpy as np
from PIL import Image

# # LOG
# import logging
# from logging_config import setup_logging
# setup_logging()

print("Welcome To Main Page")

st.set_page_config(
    page_title="Yoga App",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# CSS
local_css("style.css")


# Front-end

st.title("Yoga")

cols = st.columns(2)
with cols[0]:
    all_pose_btn = st.button("전체 자세", use_container_width=True)
    if all_pose_btn:
        st.switch_page("pages/all.py")
with cols[1]:
    custom_btn = st.button("맞춤 프로그램", icon="🧘", use_container_width=True)
    if custom_btn:
        st.switch_page("pages/custom.py")