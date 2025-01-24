import cv2
import os
import re
import mediapipe as mp
import time
import logging
from matplotlib import pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# drawing design
landmark_drawing_spec_green = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=8)
landmark_drawing_spec_red = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=8)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=6)
landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()

# parameters
start_time = None
previous_time = 0  # start time
previous_landmarks = None
movement = 0
landmark_threshold = 0.06
body_threshold = 0.8*33
no_movement_list = []  # เก็บจุดที่ไม่ขยับ
time_count = 0
time_count_threshold = 5 # stay still 5 sec
time_list = []

is_paused = False

cap = cv2.VideoCapture('test1.mp4')

def convert_ms_to_minutes(ms):
    sec = ms / 1000  
    minutes = int(sec // 60) 
    seconds = int(sec % 60)   
    return round(minutes + seconds / 60, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:

        # every 1 sec
        if current_time - previous_time >= 1000:
            previous_time = current_time

                # each landmark
            for i, landmark in enumerate(results.pose_landmarks.landmark):

                # only body and nose
                if i > 0 and i <=10 :
                    landmark.visibility = 0

                # movement calculation
                if previous_landmarks:
                    prev_landmark = previous_landmarks.landmark[i]
                    movement = ((landmark.x - prev_landmark.x)**2 +
                                (landmark.y - prev_landmark.y)**2 +
                                (landmark.z - prev_landmark.z)**2) ** 0.5  # Euclidean Distance

                # no movement
                if movement < landmark_threshold:
                    landmark_drawing_spec[i] = landmark_drawing_spec_green
                    no_movement_list.append(i)
                else:
                    landmark_drawing_spec[i] = landmark_drawing_spec_red
                    #print(f"landmark: {i} movement:  {movement}")

            print("number of stationary points: ", len(no_movement_list))
            if len(no_movement_list) >= body_threshold:
                time_count += 1
                print("no move_time count: ", time_count)
                if start_time is None:
                    start_time = convert_ms_to_minutes(cap.get(cv2.CAP_PROP_POS_MSEC))
                    print("start_time: ", start_time)              
            else:
                print("Moving")
                # no movement < 80%  -> Reset
                time_count = 0
                start_time = None
                # continue

            no_movement_list =[]

            previous_landmarks = results.pose_landmarks

        elif current_time < 1000:

            for i, landmark in enumerate(results.pose_landmarks.landmark):

                # only body and nose
                if i > 0 and i <=10 :
                    landmark.visibility = 0

                landmark_drawing_spec[i] = landmark_drawing_spec_green

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                landmark_drawing_spec=landmark_drawing_spec,
                                                connection_drawing_spec=connection_drawing_spec)
     
    if time_count >= time_count_threshold and not start_time in time_list:
        time_list.append(start_time)
        print("Saved! ", start_time)
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  # กด space bar อีกครั้งเพื่อเล่นต่อ
                break
        
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    ### video showing for MY MAC
    cv2.putText(frame, f'Time: {convert_ms_to_minutes(current_time)}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# แสดงผลเวลาที่ไม่มีการเคลื่อนไหวเกิน 5 วินาที
print("Time List:", time_list)

# ปิดการเชื่อมต่อ
cap.release()
cv2.destroyAllWindows()


# cap = cv.VideoCapture("./video.mp4")
# fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
# frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# duration = frame_count/fps

# print('fps = ' + str(fps))
# print('number of frames = ' + str(frame_count))
# print('duration (S) = ' + str(duration))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

# cap.release()
