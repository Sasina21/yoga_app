import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import os
import random
import math

selected_landmarks = [11, 12, 13, 14, 23, 24, 25, 26, 33]

left_arm = [11, 13, 15]
left_upper_arm = [11, 13]
right_arm = [12, 14, 16]
right_upper_arm = [12, 14]
left_leg = [23, 25, 27]
right_leg = [24, 26, 28]
body = [11, 12, 23, 24]

critical_points = {
    0: left_arm + right_leg + left_leg + right_leg,     #Downdog
    1: left_leg + right_leg,                            #Goddess
    2: left_arm + right_leg + left_leg + right_leg ,     #Plank
    3: left_arm,                                        #Side Plank
    4: left_leg + right_leg,                            #Tree
    5: left_leg + right_leg                             #Warrior2
}

# Load image
# filename = 'prelim/DATASET1/dataset/Tree/tree180.jpg'
filename = '/Users/ngunnnn/Downloads/ccd06167.jpg'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

base_folders = "prelim/DATASET1/dataset"

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

def calculate_twist_angle(landmarks):

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # คำนวณจุดกลางของไหล่และสะโพก
    mid_shoulder = {
        'x': (left_shoulder.x + right_shoulder.x) / 2,
        'y': (left_shoulder.y + right_shoulder.y) / 2,
        'z': (left_shoulder.z + right_shoulder.z) / 2
    }
    mid_hip = {
        'x': (left_hip.x + right_hip.x) / 2,
        'y': (left_hip.y + right_hip.y) / 2,
        'z': (left_hip.z + right_hip.z) / 2
    }
    # z_difference = abs(left_shoulder.z - right_shoulder.z)
    # print(f"มุมไหล่ซ้ายขวา{z_difference}")
    # z_difference = abs(left_hip.z - right_hip.z)
    # print(f"มุมโพกซ้ายขวา{z_difference}")
    # # เวกเตอร์ลำตัว
    torso_vector = [
        mid_shoulder['x'] - mid_hip['x'],
        mid_shoulder['y'] - mid_hip['y'],
        mid_shoulder['z'] - mid_hip['z']
    ]

    # เวกเตอร์สะโพก
    hip_vector = [
        right_hip.x - left_hip.x,
        right_hip.y - left_hip.y,
        right_hip.z - left_hip.z
    ]

    # คำนวณมุมระหว่างเวกเตอร์
    dot_product = sum(a * b for a, b in zip(torso_vector, hip_vector))
    torso_magnitude = math.sqrt(sum(a**2 for a in torso_vector))
    hip_magnitude = math.sqrt(sum(b**2 for b in hip_vector))

    if torso_magnitude == 0 or hip_magnitude == 0:
        return 0

    cos_angle = dot_product / (torso_magnitude * hip_magnitude)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp ค่า
    twist_angle = math.degrees(math.acos(cos_angle))

    return twist_angle, mid_shoulder['x'], mid_shoulder['y'], mid_shoulder['z']

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
# ฟังก์ชันโหลดจาก JSON
def load_graphs_from_json(json_path):
    with open(json_path, "r") as f:
        graph_data_list = json.load(f)

    graphs = []
    for graph_data in graph_data_list:
        G = nx.Graph()
        G.graph['filename'] = graph_data['filename']
        G.graph['classification'] = graph_data['classification']

        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})

        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **{k: v for k, v in edge.items() if k not in ['source', 'target']})

        graphs.append(G)
    return graphs

# ฟังก์ชันโหลดจาก Pickle
def load_graphs_from_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def plot_random_graph_with_image(graphs, filename):
    # เลือกกราฟสุ่ม
    # random_graph = random.choice(graphs)
    # filename = random_graph.graph['filename']
    # classification = random_graph.graph['classification']

    # # สร้าง path ไปยังภาพ
    # file_path = os.path.join(base_folder, classification, filename)
    target_graph = None
    for graph in graphs:
        if graph.graph['filename'] == os.path.basename(filename):
            target_graph = graph
            break

    # โหลดภาพ
    image = cv2.imread(filename)
    height, width, _ = image.shape

    # วาด landmarks บนภาพ
    for node_id, node_data in target_graph.nodes(data=True):
        # ตรวจสอบว่า x, y เป็น normalized หรือ absolute
        x = int(node_data['x'] * width if node_data['x'] <= 1 else node_data['x'])
        y = int(node_data['y'] * height if node_data['y'] <= 1 else node_data['y'])

        # วาดจุด landmark
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, str(node_id), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # วาด angle หากมี
        if 'angle' in node_data:
            angle = node_data['angle']
            cv2.putText(image, f"{angle:.1f}°", (x - 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if node_id in selected_landmarks:
                print(f"angle landmarks{node_id}: {angle:.1f}")

    # วาด connections จาก edges
    for u, v, edge_data in target_graph.edges(data=True):
        start_x = int(target_graph.nodes[u]['x'] * width if target_graph.nodes[u]['x'] <= 1 else target_graph.nodes[u]['x'])
        start_y = int(target_graph.nodes[u]['y'] * height if target_graph.nodes[u]['y'] <= 1 else target_graph.nodes[u]['y'])
        end_x = int(target_graph.nodes[v]['x'] * width if target_graph.nodes[v]['x'] <= 1 else target_graph.nodes[v]['x'])
        end_y = int(target_graph.nodes[v]['y'] * height if target_graph.nodes[v]['y'] <= 1 else target_graph.nodes[v]['y'])
        weight = edge_data.get('weight', 0)  # ระยะทาง (weight)

        # วาดเส้นระหว่าง landmarks
        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # แสดงระยะ (weight) บนเส้น
        mid_x, mid_y = (start_x + end_x) // 2, (start_y + end_y) // 2
        cv2.putText(image, f"{weight:.1f}", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # แสดงภาพ
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Filename: {filename}")
    plt.axis("off")
    plt.show()

def check_side_plank_side(landmarks):
    # ดึงค่าตำแหน่งสำคัญ
    left_hip = landmarks["LEFT_HIP"]
    right_hip = landmarks["RIGHT_HIP"]
    left_shoulder = landmarks["LEFT_SHOULDER"]
    right_shoulder = landmarks["RIGHT_SHOULDER"]
    left_wrist = landmarks["LEFT_WRIST"]
    right_wrist = landmarks["RIGHT_WRIST"]

    # คำนวณความสูงสะโพก
    if left_hip[1] > right_hip[1]:
        return "Side plank ขวา (ใช้แขนขวารับน้ำหนัก)"
    elif right_hip[1] > left_hip[1]:
        return "Side plank ซ้าย (ใช้แขนซ้ายรับน้ำหนัก)"
    else:
        # ตรวจสอบตำแหน่งข้อมือเพิ่มเติม
        if left_wrist[1] < right_wrist[1]:
            return "Side plank ซ้าย (ใช้แขนซ้ายรับน้ำหนัก)"
        else:
            return "Side plank ขวา (ใช้แขนขวารับน้ำหนัก)"

def extract_keypoints_as_graphs(filename):
    # Check if the image is loaded properly
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: Unable to load image '{filename}'. Please check the file path.")
    else:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_graphs = []

        # Process image with Mediapipe Pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6) as pose:
            results = pose.process(image_rgb)

            # Check if any pose landmarks were detected
            if results.pose_landmarks:
                height, width, _ = image.shape
                G = nx.Graph()

                # Add nodes for each landmark
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    G.add_node(idx, 
                           x=landmark.x , 
                           y=landmark.y , 
                           z=landmark.z)

                for connection in CUSTOM_POSE_CONNECTIONS:
                    start_idx, mid_idx = connection
                    if start_idx < len(results.pose_landmarks.landmark) and mid_idx < len(results.pose_landmarks.landmark):
                        landmark1 = results.pose_landmarks.landmark[start_idx]
                        landmark2 = results.pose_landmarks.landmark[mid_idx]

                        #Distance
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
                # twist_angle, mid_x, mid_y, mid_z = calculate_twist_angle(results.pose_landmarks.landmark)
                # G.add_node(33, 
                #         x = mid_x, 
                #         y = mid_y, 
                #         z = mid_z,
                #         angle = twist_angle,
                #         crit = 1 if twist_angle > 5 else 0
                #         )

                output_graphs.append(G)
                    

                # DRAWING
                for node_id, node_data in G.nodes(data=True):
                    # ตรวจสอบว่า x, y เป็น normalized หรือ absolute
                    # x = int(node_data['x'] * width if node_data['x'] <= 1 else node_data['x'])
                    # y = int(node_data['y'] * height if node_data['y'] <= 1 else node_data['y'])
                    x = int(node_data['x']* width)
                    y = int(node_data['y']* height)

                    # วาดจุด landmark
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(image, str(node_id), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # วาด angle หากมี
                    if 'angle' in node_data:
                        angle = node_data['angle']
                        cv2.putText(image, f"{angle:.1f}°", (x - 20, y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        if node_id in selected_landmarks:
                            print(f"angle landmarks{node_id}: {angle:.1f}")
                    
                for edge in G.edges:
                    start_node = G.nodes[edge[0]]
                    end_node = G.nodes[edge[1]]
                    cv2.line(image, 
                            (int(start_node['x']* width), int(start_node['y']* height)), 
                            (int(end_node['x']* width), int(end_node['y']* height)), 
                            (255, 0, 0), 2)

                # # Draw pose landmarks on the image
                # annotated_image = image.copy()
                # mp_drawing.draw_landmarks(
                #     annotated_image,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                # )

                # Display the annotated image
                cv2.imshow(filename, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #     # Optionally save the annotated image
            #     cv2.imwrite('00_annotated.jpg', annotated_image)
            #     print("Pose detection complete. Saved annotated image as '00_annotated.jpg'.")
            # else:
            #     print("No pose landmarks detected in the image.")
            



# โหลดจาก JSON
# train_graphs = load_graphs_from_json("svm_datatrain.json")

# หรือโหลดจาก Pickle
# train_graphs = load_graphs_from_pickle("train_graphs.pkl")

# เรียกฟังก์ชันเพื่อพลอต
# plot_random_graph_with_image(train_graphs, filename)

extract_keypoints_as_graphs(filename)
