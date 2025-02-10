import streamlit as st
import json
import os
import glob
from process import process_images

def load_results():
    OUTPUT_FILE = "storage.json"

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            return json.load(f)
    return {}

def get_image_count():
    IMAGE_FOLDER = "Images/References"
    return len(glob.glob(os.path.join(IMAGE_FOLDER, "*.*")))

def main():
    st.title("🧘‍♂️All Yoga Poses")

    results = load_results()
    poses = sorted(set(results.values()))
    
    current_image_count = get_image_count()
    if len(results) != current_image_count:
        st.warning("데이터를 업데이트하는 중입니다...")
        results = process_images() 
        st.experimental_rerun() 


    for pose in poses:
        st.markdown(f"👉 [**{pose}**](template_pose?pose={pose})")

if __name__ == "__main__":
    main()
