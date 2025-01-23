import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def process_pose_image(image_np: np.ndarray) -> np.ndarray:

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                img_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
            )

    annotated_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)

    return annotated_img
