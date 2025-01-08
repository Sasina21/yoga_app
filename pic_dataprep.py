import cv2
import mediapipe as mp

# โหลด Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # static_image_mode = True สำหรับรูปภาพ
mp_drawing = mp.solutions.drawing_utils

# โหลดรูปภาพ
image_path = "/Users/ngunnnn/Documents/thesis/yoga_app/prelim/data/sideplank/DSC_6828 copy.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ประมวลผลภาพ
results = pose.process(image_rgb)

# ตรวจสอบว่าพบ keypoints หรือไม่
if results.pose_landmarks:
    print("พบ keypoints:")
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")
    
    # วาด keypoints บนภาพ
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    # แสดงภาพที่วาด keypoints
    cv2.imshow('Pose Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ไม่พบ keypoints")

# ปิด Mediapipe
pose.close()
