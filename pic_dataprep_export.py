# import os
# import json
# import cv2
# import pandas as pd
# import mediapipe as mp
# import math

# # กำหนดโฟลเดอร์และไฟล์ output
# input_folders = [
#     "prelim/_dataset/TRAIN/Downdog",        # Classification = 0
#     "prelim/_dataset/TRAIN/Goddess",        # Classification = 1
#     "prelim/_dataset/TRAIN/Plank",          # Classification = 2
#     "prelim/_dataset/TRAIN/Side Plank",     # Classification = 3
#     "prelim/_dataset/TRAIN/Tree",           # Classification = 4
#     "prelim/_dataset/TRAIN/Warrior"         # Classification = 5
# ]
# output_csv = "output_keypoints_with_edges_angles.csv"
# output_json = "output_keypoints_with_edges_angles.json"

# # keypoints in this is set unImportant points  0
# face = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# set_labels = {
#     0: face,
#     1: face + [13, 14, 15, 16],
#     2: face + [15, 16],
#     3: face + [15, 16], ##can flip
#     4: face + [13, 14, 15, 16],
#     5: face
# }

# # ตั้งค่า Mediapipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# def save_to_csv_and_json(data, csv_path, json_path):
#     df = pd.DataFrame(data)
#     df = df[['filename'] + [col for col in df.columns if col != 'filename']]
#     df.to_csv(csv_path, index=False)

#     with open(json_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4)

# def calculate_distance(x1, y1, x2, y2, z1=0, z2=0):
#     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# def calculate_angle(p1, p2, p3):
#     ab = [p2[i] - p1[i] for i in range(3)]
#     bc = [p3[i] - p2[i] for i in range(3)]

#     dot_product = sum(ab[i] * bc[i] for i in range(3))
#     magnitude_ab = math.sqrt(sum(ab[i]**2 for i in range(3)))
#     magnitude_bc = math.sqrt(sum(bc[i]**2 for i in range(3)))

#     if magnitude_ab == 0 or magnitude_bc == 0:
#         return 0.0

#     cos_angle = dot_product / (magnitude_ab * magnitude_bc)
#     cos_angle = max(-1.0, min(1.0, cos_angle))
#     return math.degrees(math.acos(cos_angle))

# def normalize_keypoints(keypoints, image_width, image_height):
#     normalized_keypoints = {}
#     for key, value in keypoints.items():
#         if 'x_' in key:
#             normalized_keypoints[key] = value / image_width
#         elif 'y_' in key:
#             normalized_keypoints[key] = value / image_height
#         elif 'z_' in key or 'visibility_' in key or 'label_' in key:
#             normalized_keypoints[key] = value
#     return normalized_keypoints

# def extract_keypoints_from_folders(folders, output_csv, output_json):
#     data = []

#     for class_idx, folder in enumerate(folders):
#         # ใช้ set_label สำหรับ classification ที่เกี่ยวข้อง
#         set_label = set_labels.get(class_idx, [])

#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)

#             if filename.lower().endswith(('png', 'jpg', 'jpeg')):
#                 image = cv2.imread(file_path)
#                 if image is None:
#                     continue

#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 results = pose.process(image_rgb)

#                 if results.pose_landmarks:
#                     height, width, _ = image.shape
#                     keypoints = {}
#                     edges = {}
#                     angles = {}

#                     for idx, landmark in enumerate(results.pose_landmarks.landmark):
#                         keypoints[f'x_{idx}'] = landmark.x * width
#                         keypoints[f'y_{idx}'] = landmark.y * height
#                         keypoints[f'z_{idx}'] = landmark.z
#                         keypoints[f'label_{idx}'] = 0 if idx in set_label else 1

#                     # คำนวณ edges ตาม mp_pose.POSE_CONNECTIONS
#                     for connection in POSE_CONNECTIONS:
#                         start_idx, end_idx = connection
#                         if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
#                             landmark1 = results.pose_landmarks.landmark[start_idx]
#                             landmark2 = results.pose_landmarks.landmark[end_idx]
#                             distance = calculate_distance(
#                                 landmark1.x * width, landmark1.y * height, 
#                                 landmark2.x * width, landmark2.y * height, 
#                                 landmark1.z, landmark2.z
#                             )
#                             edges[f'edge_{start_idx}_{end_idx}'] = distance

#                     # # คำนวณ angles
#                     # for i in range(len(POSE_CONNECTIONS) - 1):
#                     #     if i + 1 < len(POSE_CONNECTIONS):
#                     #         p1_idx, p2_idx = POSE_CONNECTIONS[i]
#                     #         _, p3_idx = POSE_CONNECTIONS[i + 1]
#                     #         if p1_idx < len(results.pose_landmarks.landmark) and p2_idx < len(results.pose_landmarks.landmark) and p3_idx < len(results.pose_landmarks.landmark):
#                     #             p1 = (results.pose_landmarks.landmark[p1_idx].x * width,
#                     #                   results.pose_landmarks.landmark[p1_idx].y * height,
#                     #                   results.pose_landmarks.landmark[p1_idx].z)
#                     #             p2 = (results.pose_landmarks.landmark[p2_idx].x * width,
#                     #                   results.pose_landmarks.landmark[p2_idx].y * height,
#                     #                   results.pose_landmarks.landmark[p2_idx].z)
#                     #             p3 = (results.pose_landmarks.landmark[p3_idx].x * width,
#                     #                   results.pose_landmarks.landmark[p3_idx].y * height,
#                     #                   results.pose_landmarks.landmark[p3_idx].z)
#                     #             angle = calculate_angle(p1, p2, p3)
#                     #             angles[f'angle_{p1_idx}_{p2_idx}_{p3_idx}'] = angle

#                     normalized_keypoints = normalize_keypoints(keypoints, width, height)
#                     normalized_keypoints.update(edges)
#                     # normalized_keypoints.update(angles)
#                     normalized_keypoints['filename'] = filename
#                     normalized_keypoints['classification'] = class_idx
#                     data.append(normalized_keypoints)

#     save_to_csv_and_json(data, output_csv, output_json)

# extract_keypoints_from_folders(input_folders, output_csv, output_json)
# pose.close()
# print("การประมวลผลเสร็จสมบูรณ์!")

import networkx as nx
import pickle
import os
import cv2
import mediapipe as mp
import math
import json

face = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
set_labels = {
    0: face,
    1: face + [13, 14, 15, 16],
    2: face + [15, 16],
    3: face + [15, 16],  # can flip
    4: face + [13, 14, 15, 16],
    5: face
}

input_folders = [
    "prelim/_dataset/TRAIN/Downdog",        # Classification = 0
    "prelim/_dataset/TRAIN/Goddess",        # Classification = 1
    "prelim/_dataset/TRAIN/Plank",          # Classification = 2
    "prelim/_dataset/TRAIN/Side Plank",     # Classification = 3
    "prelim/_dataset/TRAIN/Tree",           # Classification = 4
    "prelim/_dataset/TRAIN/Warrior"         # Classification = 5
]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

def save_graphs_to_json(graphs, json_path):
    graph_list = []
    for G in graphs:
        graph_data = {
            "classification": G.graph.get("classification"),
            "filename": G.graph.get("filename"),
            "nodes": [
                {"id": node, **data} for node, data in G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data} for u, v, data in G.edges(data=True)
            ]
        }
        graph_list.append(graph_data)
    
    with open(json_path, 'w') as json_file:
        json.dump(graph_list, json_file, indent=4)

def calculate_distance(x1, y1, x2, y2, z1=0, z2=0):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def save_graphs_to_pickle(graphs, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(graphs, f)

def extract_keypoints_as_graphs(folders, output_pickle):
    graphs = []  # เก็บกราฟทั้งหมดในรูปแบบ List

    for class_idx, folder in enumerate(folders):
        set_label = set_labels.get(class_idx, [])

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image = cv2.imread(file_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    height, width, _ = image.shape
                    G = nx.Graph()  # สร้างกราฟใหม่สำหรับแต่ละภาพ
                    
                    G.graph['classification'] = class_idx

                    G.graph['filename'] = filename

                    # เพิ่มโหนดในกราฟ
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        G.add_node(idx, 
                                   x=landmark.x * width, 
                                   y=landmark.y * height, 
                                   z=landmark.z,
                                   label=0 if idx in set_label else 1)

                    # เพิ่มขอบในกราฟ
                    for connection in POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
                            landmark1 = results.pose_landmarks.landmark[start_idx]
                            landmark2 = results.pose_landmarks.landmark[end_idx]
                            distance = calculate_distance(
                                landmark1.x * width, landmark1.y * height, 
                                landmark2.x * width, landmark2.y * height, 
                                landmark1.z, landmark2.z
                            )
                            G.add_edge(start_idx, end_idx, weight=distance)

                    # เพิ่มกราฟ G ลงใน List
                    graphs.append(G)

    # บันทึก List ของกราฟทั้งหมดในไฟล์ pickle
    save_graphs_to_pickle(graphs, output_pickle)
    save_graphs_to_json(graphs, output_json)

output_json = "output_graphs.json"
output_pickle = "output_graphs.pkl"
extract_keypoints_as_graphs(input_folders, output_pickle)
pose.close()
print("การประมวลผลเสร็จสมบูรณ์! Export กราฟทั้งหมดเป็น pickle เรียบร้อย")


