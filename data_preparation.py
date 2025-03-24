import networkx as nx
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from pose_math import PoseMath

print(f"Welcome to DATA_PREPARATION")

JOINT_SIZE = 2

body = {11, 12, 23, 24}
forearm = {13, 14, 15, 16}
upper_arm = {11, 12, 13, 14}
thigh = {23, 24, 25, 26}
lower_leg = {25, 26, 27, 28}
shoulder = {11, 12}
hip = {23, 24}
knee = {25, 26}
wrist = {15, 16}
ankle = {27, 28}

MUSCLE_LABELS = ["gast", "vl", "gm", "ra", "es", "ld", "lt"]
MUSCLE_LABEL_MAP = {
    "salutation":   [0, 0, 0, 0, 0, 0, 0],
    "raisedarms":   [0, 0, 0, 0, 1, 1, 1],
    "handtofoot":   [0, 0, 0, 1, 1, 0 ,0],
    "equestrian":   [0, 0, 0, 0, 1, 0, 0],
    "downdog":      [0, 0, 0, 0, 1, 1, 0],
    "eightlimbed":  [0, 0, 0, 0, 1, 1, 1],
    "cobra":        [0, 0, 0, 0, 1, 1, 1],
}

MUSCLE_EDGE_MAP = {
    "gast": [(25,27), (26,28)],     # กล้ามเนื้อน่อง (Gastrocnemius - GAST)
    "vl":   [(23,25), (24,26)],     # กล้ามเนื้อต้นขาด้านนอก (Vastus lateralis - VL)
    "gm":   [(23,24)],              # กล้ามเนื้อก้น (Gluteus maximus - GM)
    "ra":   [(11,23), (12,24)],     # กล้ามเนื้อหน้าท้อง (Rectus abdominis - RA)
    "es":   [(11,23), (12,24)],     # กล้ามเนื้อแนวกระดูกสันหลัง (Erector spinae - ES)
    "ld":   [(11,23), (12,24)],     # กล้ามเนื้อหลังด้านข้าง (Latissimus dorsi - LD)
    "lt":   [(11,12)],              # กล้ามเนื้อสะบักล่าง (Lower trapezius - LT)
}

KEY_AREA = {
    "goddess": [body, hip, thigh],
    "triangle": [body, shoulder] ,
    "staff": [body],
    "catcow": [body],
    "child": [body, shoulder, hip, thigh],
    # "downdog": [body, shoulder, upper_arm, forearm, hip, lower_leg, ankle],
    "seatedforward": [body, ankle],
    "bridge": [body, hip],
    "wheel": [body, shoulder, forearm, upper_arm, hip, thigh, lower_leg],
    "halfspinaltwist": [body],
    "shoulderstand": [body, shoulder],
    "happybaby": [hip, thigh],
    "reclining": [body, hip] ,
    "cobra": [body] ,
    "plank": [body, shoulder, forearm, upper_arm, wrist],
    "warrior2": [body, thigh, shoulder],
    "chair": [body, shoulder, upper_arm, thigh],
    "mountain": [],
    "extendedmountain": [shoulder, upper_arm],
}

def save_graphs_to_json(graphs, json_path):
    graph_list = []
    for G in graphs:
        graph_data = {
            "classification": G.graph.get("classification"),
            "filename": G.graph.get("filename"),
            "label": G.graph.get("label"), 
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

# def get_file(graphs, train_pickle, test_pickle, train_json, test_json):
#     random.shuffle(graphs)
#     split_idx = int(len(graphs) * 0.75)
#     train_graphs, test_graphs = graphs[:split_idx], graphs[split_idx:]

#     # save_graphs_to_pickle(train_graphs, train_pickle)
#     # save_graphs_to_pickle(test_graphs, test_pickle)
#     save_graphs_to_json(train_graphs, train_json)
#     save_graphs_to_json(test_graphs, test_json)
    
#     print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

def get_file(graphs, train_pickle, test_pickle, train_json, test_json):

    classification_groups = {}
    for graph in graphs:
        classification = graph.graph.get('classification') 
        if classification is None:
            print("Warning: graph no 'classification'")
            continue

        if classification not in classification_groups:
            classification_groups[classification] = []
        classification_groups[classification].append(graph)
    
    train_graphs = []
    test_graphs = []

    for classification, class_graphs in classification_groups.items():
        random.shuffle(class_graphs)
        split_idx = int(len(class_graphs) * 0.75)
        train_graphs.extend(class_graphs[:split_idx])
        test_graphs.extend(class_graphs[split_idx:])

    # save_graphs_to_pickle(train_graphs, train_pickle)
    # save_graphs_to_pickle(test_graphs, test_pickle)
    save_graphs_to_json(train_graphs, train_json)
    save_graphs_to_json(test_graphs, test_json)
    
    def count_classifications(graph_list):
        counts = {}
        for graph in graph_list:
            classification = graph.graph.get('classification', 'Unknown')
            counts[classification] = counts.get(classification, 0) + 1
        return counts

    train_counts = count_classifications(train_graphs)
    test_counts = count_classifications(test_graphs)
    
    print(f"\nTrain graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

    print("\nTrain classification distribution:")
    for classification, count in train_counts.items():
        print(f"  {classification}: {count}")

    print("\nTest classification distribution:")
    for classification, count in test_counts.items():
        print(f"  {classification}: {count}")

def get_landmarks(image: np.ndarray, filename=None):

    mp_pose = mp.solutions.pose
    
    if image is None:
        print("Error: Image is None :{filename}")
        return None
    
    # .png
    if image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z} 
            for lm in results.pose_landmarks.landmark
        ]

    if landmarks is None:
        print("Error: No landmarks detected.")
        return None
        
    return landmarks

def get_key_area(graph, classification):

    if graph is None:
        print("Error: graph is None")
        return None

    for start_idx, end_idx in PoseMath.CUSTOM_POSE_CONNECTIONS:
        graph.edges[start_idx, end_idx]['key'] = 0
        for landmark_set in KEY_AREA[classification]:
            if  len(landmark_set) > JOINT_SIZE:
                if start_idx in landmark_set and end_idx in landmark_set:
                    graph.edges[start_idx, end_idx]['key'] = 1
                    break

    for idx in PoseMath.CUSTOM_JOINT:
        graph.nodes[idx]['key'] = 0
        for landmark_set in KEY_AREA[classification]:
            if  len(landmark_set) == JOINT_SIZE:
                if idx in landmark_set:
                    graph.nodes[idx]['key'] = 1
                    break

    return graph

def get_label(graph, classification):
    
    if graph is None:
        print("Error: graph is None")
        return None
    
    if classification in MUSCLE_LABEL_MAP:
        label = MUSCLE_LABEL_MAP[classification]
    else:
        label = [0] * len(MUSCLE_LABELS)

    graph.graph['label'] = label

    return graph

def get_muscle_location(graph):
    if graph is None:
        print("Error: graph is None")
        return None

    num_muscles = len(MUSCLE_LABELS)

    for u, v, attr in graph.edges(data=True):
        # เตรียม multi-hot vector สำหรับกล้ามเนื้อ
        muscle_vec = [0] * num_muscles

        for idx, muscle in enumerate(MUSCLE_LABELS):
            edge_list = MUSCLE_EDGE_MAP[muscle]

            # เช็คว่า edge (u,v) หรือ (v,u) อยู่ในตำแหน่งกล้ามเนื้อมัดนี้ไหม
            if (u, v) in edge_list or (v, u) in edge_list:
                muscle_vec[idx] = 1

        # ใส่ค่า muscle_vec เข้าไปใน edge attribute
        attr["muscle_location"] = muscle_vec

    return graph


def relabel_graph_sequentially(G):
    # Step 1: สร้าง mapping จาก id เก่า → ใหม่ (เรียงจากน้อยไปมาก)
    sorted_old_ids = sorted(G.nodes)
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_old_ids)}

    # Step 2: ใช้ NetworkX function เพื่อรีแมป node id
    G_relabel = nx.relabel_nodes(G, id_map, copy=True)

    return G_relabel

def get_graph(landmarks, classification=None, filename=None):
    if landmarks is None:
        print("Error: Landmarks are None")
        return None

    G = nx.Graph()
    G.graph['filename'] = filename
    G.graph['classification'] = classification

    # Node
    for idx, lm in enumerate(landmarks):
            G.add_node(idx,
                        x=lm['x'], 
                        y=lm['y'], 
                        z=lm['z'])
            

    for connection in PoseMath.CUSTOM_POSE_CONNECTIONS:
        start_idx, mid_idx = connection
        if start_idx < len(landmarks) and mid_idx < len(landmarks):
            lm1 = landmarks[start_idx]
            lm2 = landmarks[mid_idx]
            
            # # Add nodes only if they're involved in a connection
            # if start_idx not in G.nodes:
            #     G.add_node(start_idx, x=lm1['x'], y=lm1['y'], z=lm1['z'])
            # if mid_idx not in G.nodes:
            #     G.add_node(mid_idx, x=lm2['x'], y=lm2['y'], z=lm2['z'])

            # Distance
            distance = PoseMath.calculate_distance(
                lm1['x'], lm1['y'], 
                lm2['x'], lm2['y'], 
                lm1['z'], lm2['z']
            )
            # Direction
            direction = PoseMath.calculate_direction(
                {'x': lm1['x'], 'y': lm1['y'], 'z': lm1['z']},
                {'x': lm2['x'], 'y': lm2['y'], 'z': lm2['z']}
            )

            G.add_edge(start_idx, mid_idx, distance = distance, dir = direction)
            
            #Angle
            related_connections = [conn for conn in PoseMath.CUSTOM_POSE_CONNECTIONS if conn[0] == mid_idx or conn[0] == start_idx]

            for first, end_idx in related_connections:
                if end_idx < len(landmarks) and end_idx != start_idx and end_idx != mid_idx:
                    lm3 = landmarks[end_idx]
                    if first == mid_idx:
                        # start -> mid -> end
                        angle = PoseMath.calculate_angle(
                        {'x': lm1['x'], 'y': lm1['y'], 'z': lm1['z']},  # start_idx
                        {'x': lm2['x'], 'y': lm2['y'], 'z': lm2['z']},  # mid_idx
                        {'x': lm3['x'], 'y': lm3['y'], 'z': lm3['z']},  # end_idx
                    )
                        G.nodes[mid_idx]['angle'] = angle

                    else:
                        # mid -> start -> end
                        angle = PoseMath.calculate_angle(
                            {'x': lm2['x'], 'y': lm2['y'], 'z': lm2['z']},  # mid_idx
                            {'x': lm1['x'], 'y': lm1['y'], 'z': lm1['z']},  # start_idx
                            {'x': lm3['x'], 'y': lm3['y'], 'z': lm3['z']},  # end_idx
                        )
                        G.nodes[start_idx]['angle'] = angle
    
    # Landmark(11-12) and (23-24)
    lm11 = landmarks[11]
    lm12 = landmarks[12]
    lm23 = landmarks[23]
    lm24 = landmarks[24]

    # Direction
    direction_shoulder = PoseMath.calculate_direction(
        {'x': lm11['x'], 'y': lm11['y'], 'z': lm11['z']},
        {'x': lm12['x'], 'y': lm12['y'], 'z': lm12['z']}
    )
    direction_waist = PoseMath.calculate_direction(
        {'x': lm23['x'], 'y': lm23['y'], 'z': lm23['z']},
        {'x': lm24['x'], 'y': lm24['y'], 'z': lm24['z']}
    )
    
    # Distance
    distance_shoulder = PoseMath.calculate_distance(
        lm11['x'], lm11['y'], 
        lm12['x'], lm12['y'], 
        lm11['z'], lm12['z']
    )
    distance_waist = PoseMath.calculate_distance(
        lm23['x'], lm23['y'], 
        lm24['x'], lm24['y'], 
        lm23['z'], lm24['z']
    )
    G.add_edge(11, 12, distance=distance_shoulder, direction=direction_shoulder)
    G.add_edge(23, 24, distance=distance_waist, direction=direction_waist)
    
    if G is None:
        print("Error: Failed to generate graph from landmarks.")
        return None
    
    G = get_muscle_location(G)

    return G

def get_files_in_folders(base_folder):
    
    image_paths = {}

    for classification in os.listdir(base_folder):
        _classification = classification.lower().replace(" ", "")

        if _classification and _classification in MUSCLE_LABEL_MAP:
            classification_path = os.path.join(base_folder, classification)

            if not os.path.isdir(classification_path):
                continue
            
            if _classification not in image_paths:
                image_paths[_classification] = set()

            for filename in os.listdir(classification_path):
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):  
                    image_path = os.path.join(classification_path, filename)
                    image_paths[_classification].add(image_path)
                    

    return image_paths

if __name__ == "__main__":
    # Input folders
    base_folder = "prelim/dataset_emgref"
    # Output files
    output_train_pickle = "muslo_emg_datatrain.pkl"
    output_test_pickle = "muslo_emg_datatest.pkl"
    output_train_json = "muslo_emg_datatrain.json"
    output_test_json = "muslo_emg_datatest.json"

    graphs = []

    image_paths = get_files_in_folders(base_folder)
    
    for classification, image_paths in image_paths.items():
        for image_path in image_paths:
            image = cv2.imread(image_path)
            landmarks = get_landmarks(image)
            graph = get_graph(landmarks, classification, image_path)
            # graph = get_key_area(graph, classification)
            graph = get_label(graph, classification)

            if graph:
                graphs.append(graph)
                print(f"Processed: {image_path}")

    # extract_keypoints_as_graphs(base_folders, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
    # graphs = extract_graphs(base_folders)
    get_file(graphs, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
    print("Export success!!!!!!!!")