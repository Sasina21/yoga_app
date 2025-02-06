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

left_arm = [11, 13, 15]
left_upper_arm = [11, 13]
right_arm = [12, 14, 16]
right_upper_arm = [12, 14]
left_leg = [23, 25, 27]
right_leg = [24, 26, 28]
body = [11, 12, 23, 24]

CRITICAL_POINTS = {
    "Downdog": left_arm + right_arm + left_leg + right_leg,
    "Goddess": left_leg + right_leg,
    "Plank": left_arm + right_arm + left_leg + right_leg ,
    "Tree": left_leg + right_leg,  
    "Cobra": left_arm + right_arm + body,
    "Catcow": left_arm + right_arm + left_leg + right_leg,
    "Staff": left_leg + right_leg + body,
    "Warrior1": left_leg + right_leg,
    "Warrior2": left_leg + right_leg,

    "Sideplank": left_arm,                    
}

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

def get_file(graphs, train_pickle, test_pickle, train_json, test_json):
    random.shuffle(graphs)
    split_idx = int(len(graphs) * 0.75)
    train_graphs, test_graphs = graphs[:split_idx], graphs[split_idx:]

    # save_graphs_to_pickle(train_graphs, train_pickle)
    # save_graphs_to_pickle(test_graphs, test_pickle)
    save_graphs_to_json(train_graphs, train_json)
    save_graphs_to_json(test_graphs, test_json)
    
    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

def get_landmarks(image: np.ndarray, filename=None):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
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
        
        return landmarks

def get_graph(landmarks, classification=None, filename=None):
    if landmarks is None:
        print("Error: Landmarks are None")
        return None

    G = nx.Graph()
    G.graph['filename'] = filename


    if classification:
        G.graph['classification'] = classification
        # Node and Critical points
        for idx, lm in enumerate(landmarks):
            crit = 1 if classification and classification in CRITICAL_POINTS and idx in CRITICAL_POINTS[classification] else 0
            G.add_node(idx, 
                       x=lm['x'], 
                       y=lm['y'], 
                       z=lm['z'], 
                       crit=crit)
    else:
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

    
    # Distance landmark(11-12) and (23-24)
    lm11 = landmarks[11]
    lm12 = landmarks[12]
    lm23 = landmarks[23]
    lm24 = landmarks[24]
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
    G.add_edge(11, 12, distance=distance_shoulder)
    G.add_edge(23, 24, distance=distance_waist)
    
    return G

if __name__ == "__main__":
    # Input folders
    base_folders = "prelim/DATASET1/symmetric"
    # Output files
    output_train_pickle = "9sym_svm_datatrain.pkl"
    output_test_pickle = "9sym_svm_datatest.pkl"
    output_train_json = "9sym_svm_datatrain.json"
    output_test_json = "9sym_svm_datatest.json"

    graphs = []

    for folder_name in os.listdir(base_folders):
        classification_path = os.path.join(base_folders, folder_name)
        if not os.path.isdir(classification_path):
            continue

        for filename in os.listdir(classification_path):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):  
                image_path = os.path.join(classification_path, filename)
                image = cv2.imread(image_path)
                landmarks = get_landmarks(image)
                graph = get_graph(landmarks, folder_name, filename)
                if graph:
                    graphs.append(graph)
                    print(f"Processed: {filename}")

    # extract_keypoints_as_graphs(base_folders, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
    # graphs = extract_graphs(base_folders)
    get_file(graphs, "9sym_svm_datatrain.pkl", "9sym_svm_datatest.pkl", "9sym_svm_datatrain.json", "9sym_svm_datatest.json")
    print("Export success!!!!!!!!")