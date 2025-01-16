import networkx as nx
import pickle
import os
import cv2
import mediapipe as mp
import math
import json
import random
import matplotlib.pyplot as plt

input_folders = []
base_folders = "prelim/DATASET1/symmetric"

left_arm = [11, 13, 15]
left_upper_arm = [11, 13]
right_arm = [12, 14, 16]
right_upper_arm = [12, 14]
left_leg = [23, 25, 27]
right_leg = [24, 26, 28]
body = [11, 12, 23, 24]

critical_points = {
    "Downdog": left_arm + right_arm + left_leg + right_leg,
    "Goddess": left_leg + right_leg,
    "Plank": left_arm + right_arm + left_leg + right_leg ,
    "Tree": left_leg + right_leg,  
    "Cobra": left_arm + right_arm + body,
    "Catcow": left_arm + right_arm + left_leg + right_leg,
    "Staff": left_leg + right_leg + body,

    "Sideplank": left_arm,                    
    "Warrior2": left_leg + right_leg
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

CUSTOM_POSE_CONNECTIONS = frozenset([
    # # body
    # (11, 12),  # ไหล่ซ้ายเชื่อมกับไหล่ขวา
    # (23, 24),  # สะโพกซ้ายเชื่อมกับสะโพกขวา

    # left arm
    (11, 13), (13, 15),  
    (11, 23),

    # right arm
    (12, 14), (14, 16),  
    (12, 24),

    # left leg
    (23, 25), (25, 27),

    # right
    (24, 26), (26, 28)
])

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

def save_graphs_to_pickle(graphs, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(graphs, f)

def calculate_distance(x1, y1, x2, y2, z1=0, z2=0):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def calculate_angle(a, b, c):
    # ba = [(a['x'] - b['x']), (a['y'] - b['y']), (a['z'] - b['z'])]
    # bc = [(c['x'] - b['x']), (c['y'] - b['y']), (c['z'] - b['z'])]
    ba = [(a['x'] - b['x']), (a['y'] - b['y'])]
    bc = [(c['x'] - b['x']), (c['y'] - b['y'])]

    # คำนวณ dot product
    dot_product = sum(ba[i] * bc[i] for i in range(2))

    # คำนวณขนาด (magnitude) ของเวกเตอร์
    magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(2)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(2)))

    # ตรวจสอบกรณี magnitude เป็น 0
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    # คำนวณ cos ของมุม
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)

    # หามุมจาก cos (จำกัดค่า -1 ถึง 1)
    angle = math.acos(max(-1, min(1, cos_angle)))
    result = math.degrees(angle)
    
    return result

def calculate_direction(a, b):
    direction = {
        'x': b['x'] - a['x'],
        'y': b['y'] - a['y'],
        'z': b['z'] - a['z']
    }
    return direction

def calculate_twist_angle(landmarks):

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    shoulder_vector = [
        right_shoulder.x - left_shoulder.x,
        right_shoulder.y - left_shoulder.y,
        right_shoulder.z - left_shoulder.z
    ]
    
    hip_vector = [
        right_hip.x - left_hip.x,
        right_hip.y - left_hip.y,
        right_hip.z - left_hip.z
    ]
    
    # คำนวณ dot product และ magnitude
    dot_product = sum(a * b for a, b in zip(shoulder_vector, hip_vector))
    shoulder_magnitude = math.sqrt(sum(a**2 for a in shoulder_vector))
    hip_magnitude = math.sqrt(sum(b**2 for b in hip_vector))
    
    if shoulder_magnitude == 0 or hip_magnitude == 0:
        twist_angle = 0
    else:
        cos_angle = dot_product / (shoulder_magnitude * hip_magnitude)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp ค่าให้อยู่ในช่วง [-1, 1]
        twist_angle = math.degrees(math.acos(cos_angle)) 

    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    center_z = (left_shoulder.z + right_shoulder.z) / 2

    return twist_angle, center_x, center_y, center_z

def extract_keypoints_as_graphs(folders, train_pickle, test_pickle, train_json, test_json):
    # graphs = []
    train_graphs = []
    test_graphs = []

    for folder in os.listdir(folders):
        folder_path = os.path.join(folders, folder)
        if not os.path.isdir(folder_path):
            continue

        classification = folder
        filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        random.shuffle(filenames)
        split_idx = int(len(filenames) * 0.75)
        train_files = filenames[:split_idx]
        test_files = filenames[split_idx:]

        for dataset, output_graphs in [(train_files, train_graphs), (test_files, test_graphs)]:
            for filename in dataset:
                file_path = os.path.join(folder_path, filename)

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error loading image: {file_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Process image with Mediapipe Pose
                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                    results = pose.process(image_rgb)

                    # Check if any pose landmarks were detected
                    if results.pose_landmarks:
                        height, width, _ = image.shape
                        G = nx.Graph()
                        G.graph['filename'] = filename
                        G.graph['classification'] = classification

                        # Add nodes for each landmark
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):

                            if classification in critical_points and idx in critical_points[classification]:
                                crit = 1
                            else:
                                crit = 0

                            G.add_node(idx, 
                                x=landmark.x, 
                                y=landmark.y, 
                                z=landmark.z,
                                crit = crit
                                )
                            
                        for connection in CUSTOM_POSE_CONNECTIONS:
                            start_idx, mid_idx = connection
                            if start_idx < len(results.pose_landmarks.landmark) and mid_idx < len(results.pose_landmarks.landmark):
                                landmark1 = results.pose_landmarks.landmark[start_idx]
                                landmark2 = results.pose_landmarks.landmark[mid_idx]

                                # Distance
                                distance = calculate_distance(
                                    landmark1.x, landmark1.y, 
                                    landmark2.x, landmark2.y, 
                                    landmark1.z, landmark2.z
                                )
                                G.add_edge(start_idx, mid_idx, distance=distance)

                                # Direction
                                if connection not in [(12, 24), (11, 23)]:
                                    direction = calculate_direction(
                                        {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},
                                        {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z}
                                    )
                                    G.nodes[start_idx]['dir'] = direction
                                
                                #Angle
                                related_connections = [conn for conn in CUSTOM_POSE_CONNECTIONS if conn[0] == mid_idx or conn[0] == start_idx]

                                for first, end_idx in related_connections:
                                    if end_idx < len(results.pose_landmarks.landmark) and end_idx != start_idx and end_idx != mid_idx:
                                        landmark3 = results.pose_landmarks.landmark[end_idx]
                                        if first == mid_idx:
                                            # start -> mid -> end
                                            angle = calculate_angle(
                                            {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},  # start_idx
                                            {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},  # mid_idx
                                            {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z},  # end_idx
                                        )
                                            G.nodes[mid_idx]['angle'] = angle

                                        else:
                                            # mid -> start -> end
                                            angle = calculate_angle(
                                                {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},  # mid_idx
                                                {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},  # start_idx
                                                {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z},  # end_idx
                                            )
                                            G.nodes[start_idx]['angle'] = angle

                        
                        # Distance landmark(11-12) and (23-24)
                        landmark11 = results.pose_landmarks.landmark[11]
                        landmark12 = results.pose_landmarks.landmark[12]
                        landmark23 = results.pose_landmarks.landmark[23]
                        landmark24 = results.pose_landmarks.landmark[24]
                        distance_shoulder = calculate_distance(
                            landmark11.x, landmark11.y, 
                            landmark12.x, landmark12.y, 
                            landmark11.z, landmark12.z
                        )
                        distance_waist = calculate_distance(
                            landmark23.x, landmark23.y, 
                            landmark24.x, landmark24.y, 
                            landmark23.z, landmark24.z
                        )
                        G.add_edge(11, 12, distance=distance_shoulder)
                        G.add_edge(23, 24, distance=distance_waist)

                        # # Add LANDMARK33 for twisting pose landmark(12-13) and (23-24)
                        # twist_angle, center_x, center_y, center_z = calculate_twist_angle(results.pose_landmarks.landmark)
                        # G.add_node(33, 
                        #         x = center_x, 
                        #         y = center_y, 
                        #         z = center_z,
                        #         angle = twist_angle,
                        #         crit = 1 if twist_angle > 5 else 0
                        #         )

                        # Store the graph
                        output_graphs.append(G)

                # results = pose.process(image_rgb)

                # if results.pose_landmarks:
                #     height, width, _ = image.shape
                #     G = nx.Graph()

                #     G.graph['filename'] = filename
                #     G.graph['classification'] = classification

                #     # Node
                #     for idx, landmark in enumerate(results.pose_landmarks.landmark):
                #         G.add_node(idx, 
                #                    x=landmark.x, 
                #                    y=landmark.y, 
                #                    z=landmark.z)
                    
                #     # Distance
                #     for connection in CUSTOM_POSE_CONNECTIONS:
                #         start_idx, mid_idx = connection
                #         if start_idx < len(results.pose_landmarks.landmark) and mid_idx < len(results.pose_landmarks.landmark):
                #             landmark1 = results.pose_landmarks.landmark[start_idx]
                #             landmark2 = results.pose_landmarks.landmark[mid_idx]
                #             distance = calculate_distance(
                #                 landmark1.x, landmark1.y, 
                #                 landmark2.x, landmark2.y, 
                #                 landmark1.z, landmark2.z
                #             )
                #             G.add_edge(start_idx, mid_idx, distance=distance)

                #             # คำนวณมุมโดยใช้จุด Landmark ที่เกี่ยวข้อง
                #             # หา connection ที่มี mid_idx เป็นจุดเริ่มต้น
                #             related_connections = [conn for conn in CUSTOM_POSE_CONNECTIONS if conn[0] == mid_idx or conn[0] == start_idx]

                #             for first, end_idx in related_connections:
                #                 if end_idx < len(results.pose_landmarks.landmark) and end_idx != start_idx and end_idx != mid_idx:
                #                     landmark3 = results.pose_landmarks.landmark[end_idx]
                #                     if first == mid_idx:
                #                         # start -> mid -> end
                #                         angle = calculate_angle(
                #                         {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},  # start_idx
                #                         {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},  # mid_idx
                #                         {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z},  # end_idx
                #                     )
                #                         G.nodes[mid_idx]['angle'] = angle

                #                     else:
                #                         # mid -> start -> end
                #                         angle = calculate_angle(
                #                             {'x': landmark2.x, 'y': landmark2.y, 'z': landmark2.z},  # mid_idx
                #                             {'x': landmark1.x, 'y': landmark1.y, 'z': landmark1.z},  # start_idx
                #                             {'x': landmark3.x, 'y': landmark3.y, 'z': landmark3.z},  # end_idx
                #                         )
                #                         G.nodes[start_idx]['angle'] = angle

                #     output_graphs.append(G)

    # Save train and test graphs as both Pickle and JSON
    save_graphs_to_pickle(train_graphs, train_pickle)
    save_graphs_to_json(train_graphs, train_json)
    save_graphs_to_pickle(test_graphs, test_pickle)
    save_graphs_to_json(test_graphs, test_json)

    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")


# Output files
output_train_pickle = "sym_svm_datatrain.pkl"
output_test_pickle = "sym_svm_datatest.pkl"
output_train_json = "sym_svm_datatrain.json"
output_test_json = "sym_svm_datatest.json"

extract_keypoints_as_graphs(base_folders, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
pose.close()
print("success!!!!!!!!")