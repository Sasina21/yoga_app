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

# body = [11, 12, 23, 24]
# left_shoulder = [11]
# right_shoulder = [12]
# hip = [23, 24]
# left_forearm = [13, 15]
# left_upper_arm = [11, 13]
# right_forearm = [14, 16]
# right_upper_arm = [12, 14]
# left_thigh = [23, 25]
# left_lower_leg = [25, 27]
# right_thigh = [24, 26]
# right_lower_leg = [26, 28]
# left_wrist = [15]
# right_wrist = [16]
# left_ankle = [27]
# right_ankle = [28]
# left_knee = [25]
# right_knee = [26]

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

KEY_AREA = {
    # "Downdog": left_arm + right_arm + left_leg + right_leg,
    # "Goddess": left_leg + right_leg,
    # "Plank": left_arm + right_arm + left_leg + right_leg ,
    # "Tree": left_leg + right_leg,  
    # "Cobra": left_arm + right_arm + body,
    # "Catcow": left_arm + right_arm + left_leg + right_leg,
    # "Staff": left_leg + right_leg + body,
    # "Warrior1": left_leg + right_leg,
    # "Warrior2": left_leg + right_leg,

    "goddess": [body, hip, thigh],
    "triangle": [body, shoulder] ,
    "staff": [body],
    "catcow": [body],
    "child": [body, shoulder, hip, thigh],
    "downdog": [body, shoulder, upper_arm, forearm, hip, lower_leg, ankle],
    "seatedforward": [body, ankle],
    "bridge": [body, hip],
    "wheel": [body, shoulder, forearm, upper_arm, hip, thigh, lower_leg],
    "halfspinaltwist": [body],
    "shoulderstand": [body, shoulder],
    "happybaby": [hip, thigh],
    "reclining": [body, hip] ,
    "cobra": [body] ,
    "plank": [body, shoulder, forearm, upper_arm, wrist],
    "warrior2": [body, thigh, shoulder]

}

CRITICAL = {
    "Downdog": wrist + ankle,
    "Triangle" : body + shoulder + hip + thigh + ankle,
    "Seated Forward" : body + hip + ankle,
    "Cobra": body + shoulder, 
    "Child" : body + shoulder + hip + thigh,
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

# def get_file(graphs, train_pickle, test_pickle, train_json, test_json):
#     random.shuffle(graphs)
#     split_idx = int(len(graphs) * 0.75)
#     train_graphs, test_graphs = graphs[:split_idx], graphs[split_idx:]

#     # save_graphs_to_pickle(train_graphs, train_pickle)
#     # save_graphs_to_pickle(test_graphs, test_pickle)
#     save_graphs_to_json(train_graphs, train_json)
#     save_graphs_to_json(test_graphs, test_json)
    
#     print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

import random

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

    for idx, data in graph.nodes(data=True):
        key = 0
        if classification and classification in KEY_AREA and idx in KEY_AREA[classification]:
            key = 1
        graph.nodes[idx]['key'] = key


    for landmark_set in KEY_AREA[classification]:
        has_connection = False

        for start_idx, end_idx in PoseMath.CUSTOM_POSE_CONNECTIONS:
            if start_idx in landmark_set and end_idx in landmark_set:
                has_connection = True

                if graph.has_edge(start_idx, end_idx):
                    graph.edges[start_idx, end_idx]['key'] = 1


        if not has_connection:
            for idx in landmark_set:
                if idx in graph.nodes:
                    graph.nodes[idx]['key'] = 1

    return graph

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
    
    if G is None:
        print("Error: Failed to generate graph from landmarks.")
        return None
    
    return G

def get_files_in_folders(base_folder):
    
    image_paths = {}

    for classification in os.listdir(base_folder):
        _classification = classification.lower().replace(" ", "")

        if _classification and _classification in KEY_AREA:
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
    base_folder = "prelim/DATASET/dataset"
    # Output files
    output_train_pickle = "15classes_datatrain.pkl"
    output_test_pickle = "15classes_datatest.pkl"
    output_train_json = "15classes_datatrain.json"
    output_test_json = "15classes_datatest.json"

    graphs = []

    image_paths = get_files_in_folders(base_folder)
    
    for classification, image_paths in image_paths.items():
        for image_path in image_paths:
            image = cv2.imread(image_path)
            landmarks = get_landmarks(image)
            graph = get_graph(landmarks, classification, image_path)
            graph = get_key_area(graph, classification)

            if graph:
                graphs.append(graph)
                print(f"Processed: {image_path}")

    # extract_keypoints_as_graphs(base_folders, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
    # graphs = extract_graphs(base_folders)
    get_file(graphs, output_train_pickle, output_test_pickle, output_train_json, output_test_json)
    print("Export success!!!!!!!!")