import os
import joblib
import pandas as pd
from data_preparation import extract_graph

# Load trained model, scaler, and label encoder
best_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def predict_pose(image_path):

    # Extract graph from image
    graph = extract_graph(image_path)
    
    if not graph:
        print(f"Failed to process image: {image_path}")
        return None
    
    # Extract features
    angles = []
    dirs = []

    for node_id, node in graph.nodes(data=True):
        if "angle" in node:
            angles.append(node["angle"])

    for _, _, edge_data in graph.edges(data=True):
        if "dir" in edge_data:
            dirs.extend([edge_data["dir"]["x"], edge_data["dir"]["y"], edge_data["dir"]["z"]])
    
    print(f'Extracted {len(angles)} angles and {len(dirs)} direction values')
    
    if not angles or not dirs:
        print(f"Insufficient data extracted from image: {image_path}")
        return None
    
    # Convert to DataFrame
    feature_data = {
        **{f'angle_{i}': angle for i, angle in enumerate(angles)},
        **{f'dir_{i}': dir_val for i, dir_val in enumerate(dirs)}
    }
    
    feature_df = pd.DataFrame([feature_data])
    
    # Normalize the features using the pre-trained scaler
    print(f"Feature data shape: {feature_df.shape}")
    feature_scaled = scaler.transform(feature_df)
    
    # Predict the class
    prediction = best_model.predict(feature_scaled)
    
    # Decode label
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_class


if __name__ == "__main__":
    image_path = "prelim/DATASET1/ให้จี้/downdog.jpg"  # เปลี่ยนเป็นพาธของรูปที่ต้องการพยากรณ์
    predicted_pose = predict_pose(image_path)
    
    if predicted_pose:
        print(f"Predicted Pose: {predicted_pose}")



# with open("svm_model.joblib", 'rb') as f:
#     model = joblib.load(f)
# with open("scaler.joblib", 'rb') as f:
#     scaler = joblib.load(f)
# with open("label_encoder.joblib", 'rb') as f:
#     label_encoder = joblib.load(f)

# expected_feature_count = scaler.n_features_in_
# print(f"✅ Model expects {expected_feature_count} features.")

# def extract_pose_features(image_np):
    
#     image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
#     with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
#         results = pose.process(image_rgb)

#         if not results.pose_landmarks:
#             return None

#         landmarks = results.pose_landmarks.landmark
#         angles = []
#         dirs = []
#         computed_angles = {}

#         for connection in PoseMath.CUSTOM_POSE_CONNECTIONS:
#             start_idx, mid_idx = connection
#             if start_idx < len(results.pose_landmarks.landmark) and mid_idx < len(results.pose_landmarks.landmark):
#                 landmark1 = results.pose_landmarks.landmark[start_idx]
#                 landmark2 = results.pose_landmarks.landmark[mid_idx]

#                 # Direction
#                 direction = PoseMath.calculate_direction(
#                     {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},
#                     {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z}
#                 )

#                 dirs.extend([direction['x'], direction['y'], direction['z']])
                
#                 #Angle
#                 related_connections = [conn for conn in PoseMath.CUSTOM_POSE_CONNECTIONS if conn[0] == mid_idx or conn[0] == start_idx]

#                 for first, end_idx in related_connections:
#                     if end_idx < len(landmarks) and end_idx != start_idx and end_idx != mid_idx:
#                         landmark3 = landmarks[end_idx]

#                         if first == mid_idx:
#                             # ✅ เช็คว่ามี mid_idx ใน computed_angles หรือยัง
#                             if mid_idx in computed_angles:
#                                 continue

#                             angle = PoseMath.calculate_angle(
#                                 {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},
#                                 {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},
#                                 {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z}
#                             )
#                             computed_angles[mid_idx] = angle  # ✅ เก็บค่า angle ตาม mid_idx

#                         else:
#                             # ✅ เช็คว่า start_idx ถูกใช้ไปแล้วหรือยัง
#                             if start_idx in computed_angles:
#                                 continue  # ข้ามถ้าคำนวณไปแล้ว

#                             angle = PoseMath.calculate_angle(
#                                 {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},
#                                 {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},
#                                 {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z}
#                             )
#                             computed_angles[start_idx] = angle  # ✅ เก็บค่า angle ตาม start_idx

#                         angles.append(angle)

#         print(f"✅ Extracted Angles: {len(angles)} values")
#         print(f"✅ Extracted Directions: {len(dirs)} values")
#         print(f"✅ Total Features: {len(angles) + len(dirs)}")

#         return angles, dirs

# def predict_pose(image_path):

#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error pic loading: {image_path}")
#         return None

#     features = extract_pose_features(image)
#     if features is None:
#         print(f"keypoints not found: {image_path}")
#         return None

#     angles, dirs = features
#     feature_vector = angles + dirs
#     print(f"features: {feature_vector}")


#     # ตรวจสอบว่าจำนวนฟีเจอร์ตรงกับที่โมเดลคาดหวัง
#     if len(feature_vector) != expected_feature_count:
#         print(f"❌ Error: Feature vector มี {len(feature_vector)} ฟีเจอร์ แต่โมเดลต้องการ {expected_feature_count} ฟีเจอร์")
#         return None

#     df = pd.DataFrame([feature_vector])
#     df_scaled = scaler.transform(df)
#     prediction = model.predict(df_scaled)
#     predicted_label = label_encoder.inverse_transform(prediction)[0]

#     return predicted_label

# image_path = "prelim/DATASET1/symmetric/Goddess/goddess5.jpg"
# if os.path.exists(image_path):
#     predicted_pose = predict_pose(image_path)
#     if predicted_pose:
#         print(f"✅ file: {image_path} → Predict result: {predicted_pose}")
# else:
#     print("❌ Error: file not found")
