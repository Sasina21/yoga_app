import cv2
import mediapipe as mp

# เริ่มต้น mediapipe pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพจาก BGR เป็น RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ตรวจจับ pose
    results = pose.process(rgb_frame)

    # วาด landmark และเชื่อมจุดต่างๆ
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # แสดงผลวิดีโอ
    cv2.imshow('Pose Detection', frame)

    # ออกจากการประมวลผลเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อ
cap.release()
cv2.destroyAllWindows()

