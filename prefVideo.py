import cv2
import mediapipe as mp
import time

# เริ่มต้น mediapipe pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture('test.mp4')

# ตัวแปรเก็บตำแหน่งก่อนหน้า (ใช้สำหรับการคำนวณการเคลื่อนไหว)
previous_landmarks = None
no_movement_duration = 0  # เก็บเวลาที่ไม่มีการเคลื่อนไหว
no_movement_threshold = 0.1  # เกณฑ์การเคลื่อนไหวที่น้อย
no_movement_list = []  # เก็บเวลาที่ไม่มีการเคลื่อนไหวเกิน 5 วินาที
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพจาก BGR เป็น RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ตรวจจับ pose
    results = pose.process(rgb_frame)

    # หากมีการตรวจจับ landmark
    if results.pose_landmarks:
        # วาด landmark และเชื่อมจุดต่างๆ
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # คำนวณการเคลื่อนไหว
        movement_detected = False
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if previous_landmarks:
                prev_landmark = previous_landmarks.landmark[i]
                movement = ((landmark.x - prev_landmark.x)**2 + 
                            (landmark.y - prev_landmark.y)**2 + 
                            (landmark.z - prev_landmark.z)**2) ** 0.5  # คำนวณระยะทางการเคลื่อนไหว

                # ถ้าการเคลื่อนไหวต่ำกว่าค่าที่กำหนด
                if movement > no_movement_threshold:
                    movement_detected = True
        # ถ้าไม่พบการเคลื่อนไหว
        if not movement_detected:
            if start_time is None:  # เริ่มต้นเวลา
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # เริ่มจับเวลา (เวลาของคลิป)
        else:
            if start_time is not None:  # ถ้ามีการเคลื่อนไหวและมีการจับเวลา
                elapsed_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start_time  # คำนวณเวลาที่ไม่มีการเคลื่อนไหว
                if elapsed_time >= 5:  # ถ้าเวลาผ่านไป 5 วินาที
                    # บันทึกเวลาของคลิป (ใช้เวลาที่ผ่านไปจาก `cap.get()`)
                    no_movement_list.append(start_time)  # เก็บเวลาเริ่มต้น
                start_time = None  # รีเซ็ตเวลาเมื่อพบการเคลื่อนไหว

        # เก็บข้อมูล landmark ของครั้งล่าสุด
        previous_landmarks = results.pose_landmarks


    # # คำนวณเวลาที่คลิปเล่นไปแล้ว
    # current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # เวลาในหน่วยวินาที

    # # แสดงเวลาในหน้าจอ
    # cv2.putText(frame, f'Time: {current_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # # แสดงผลวิดีโอ
    # cv2.imshow('Pose Detection', frame)

    # # ออกจากการประมวลผลเมื่อกด 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# แสดงผลเวลาที่ไม่มีการเคลื่อนไหวเกิน 5 วินาที
print("No Movement List:", no_movement_list)

# ปิดการเชื่อมต่อ
cap.release()
cv2.destroyAllWindows()
