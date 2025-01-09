import networkx as nx
import pickle
import os
import cv2
import mediapipe as mp
import math
import json

relevant_nodes = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# critical points val=1
face = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
left_arm = [13, 15]
left_upper_arm = [13]
right_arm = [14, 16]
right_upper_arm = [14]
left_leg = [25, 27]
right_leg = [26, 28]
body = [11, 12, 23, 24]

set_labels = {
    0: body + right_arm + left_leg + right_leg, 
    1: body + left_leg + right_leg,
    2: body + right_arm + left_arm + right_leg + left_leg,
    3: body + left_upper_arm + right_upper_arm + left_leg,  
    4: body + left_upper_arm + right_upper_arm + right_leg,
    5: body + right_upper_arm + right_leg,
    6: body + left_upper_arm + left_leg,
    7: body + left_arm + right_upper_arm + left_leg + right_leg,
    8: body + left_upper_arm + right_arm + left_leg + right_leg
}

input_folders = []
base_folders = "prelim/ dataset/TRAIN" # 34 รูป
for i in range(18):
    folder_path = os.path.join(base_folders, str(i))  
    if os.path.exists(folder_path):
        print(f"Reading folder: {folder_path}")
        input_folders.append(folder_path)

# input_folders = [
#     "prelim/_dataset/TRAIN/Downdog",        # Classification = 0
#     "prelim/_dataset/TRAIN/Goddess",        # Classification = 1
#     "prelim/_dataset/TRAIN/Plank",          # Classification = 2
#     "prelim/_dataset/TRAIN/Side Plank",     # Classification = 3
#     "prelim/_dataset/TRAIN/Tree",           # Classification = 4
#     "prelim/_dataset/TRAIN/Warrior"         # Classification = 5
# ]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

def save_graphs_to_json(graphs, json_path):
    graph_list = []
    for G in graphs:
        graph_data = {
            # "classification": G.graph.get("classification"),
            "filename": G.graph.get("filename"),
            "left_arm_label": G.graph.get("left_arm_label"),
            "right_arm_label": G.graph.get("right_arm_label"),
            "left_upper_arm_label": G.graph.get("left_upper_arm_label"),
            "right_upper_arm_label": G.graph.get("right_upper_arm_label"),
            "left_leg_label": G.graph.get("left_leg_label"),
            "right_leg_label": G.graph.get("right_leg_label"),
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


def calculate_angle(a, b, c):
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    dot_product = sum(ba[i] * bc[i] for i in range(3))
    magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(3)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(3)))

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.acos(max(-1, min(1, cos_angle)))  # Clamp value to avoid math domain error
    return math.degrees(angle)


def save_graphs_to_pickle(graphs, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(graphs, f)


def extract_keypoints_as_graphs(folders, output_pickle):
    graphs = []

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
                    label_list = []
                    G = nx.Graph()

                    # G.graph['classification'] = class_idx
                    G.graph['filename'] = filename

                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        G.add_node(idx, 
                                   x=landmark.x * width, 
                                   y=landmark.y * height, 
                                   z=landmark.z)

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

                    # คำนวณมุมและเพิ่มเป็นข้อมูลในโหนด
                    for i in range(len(POSE_CONNECTIONS) - 1):
                        if i + 1 < len(POSE_CONNECTIONS):
                            p1_idx, p2_idx = POSE_CONNECTIONS[i]
                            _, p3_idx = POSE_CONNECTIONS[i + 1]
                            if p1_idx < len(results.pose_landmarks.landmark) and p2_idx < len(results.pose_landmarks.landmark) and p3_idx < len(results.pose_landmarks.landmark):
                                p1 = (results.pose_landmarks.landmark[p1_idx].x * width,
                                    results.pose_landmarks.landmark[p1_idx].y * height,
                                    results.pose_landmarks.landmark[p1_idx].z)
                                p2 = (results.pose_landmarks.landmark[p2_idx].x * width,
                                    results.pose_landmarks.landmark[p2_idx].y * height,
                                    results.pose_landmarks.landmark[p2_idx].z)
                                p3 = (results.pose_landmarks.landmark[p3_idx].x * width,
                                    results.pose_landmarks.landmark[p3_idx].y * height,
                                    results.pose_landmarks.landmark[p3_idx].z)
                                angle = calculate_angle(p1, p2, p3)
                                G.nodes[p2_idx]['angle'] = angle

                    G.graph['left_arm_label'] = (1 if all(node in set_label for node in left_arm) else 0)
                    G.graph['right_arm_label'] = (1 if all(node in set_label for node in right_arm) else 0)
                    G.graph['left_upper_arm_label'] = (1 if all(node in set_label for node in left_upper_arm) else 0)
                    G.graph['right_upper_arm_label'] = (1 if all(node in set_label for node in right_upper_arm) else 0)
                    G.graph['left_leg_label'] = (1 if all(node in set_label for node in left_leg) else 0)
                    G.graph['right_leg_label'] = (1 if all(node in set_label for node in right_leg) else 0)

                    
                    graphs.append(G)

    save_graphs_to_pickle(graphs, output_pickle)
    save_graphs_to_json(graphs, output_json)


output_json = "output_graphs.json"
output_pickle = "output_graphs.pkl"
extract_keypoints_as_graphs(input_folders, output_pickle)
pose.close()
print("การประมวลผลเสร็จสมบูรณ์! Export กราฟทั้งหมดเป็น pickle และ JSON เรียบร้อย")
