import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from PIL import Image, ImageDraw
import mediapipe as mp
import pandas as pd
import cv2
from data_preparation import get_graph, get_landmarks

print(f"Welcome to KEY_AREA_PREDICTION")

# Load trained model
model_path = "rgcn_model.pt"

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(RGCNConv, self).__init__(aggr='add')  # Sum aggregation
        self.node_linear = torch.nn.Linear(in_channels, out_channels)
        self.edge_linear = torch.nn.Linear(edge_dim, out_channels)
        self.activation = F.relu

    def forward(self, x, edge_index, edge_attr):
        # Linear transformation on nodes and edges
        x = self.node_linear(x)
        edge_attr = self.edge_linear(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Combine node and edge features
        return x_j + edge_attr

class RGCNNodeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super(RGCNNodeModel, self).__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, edge_dim)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, edge_dim)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, edge_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr).relu()
        return self.fc(x)

def draw_landmarks(image_pil, landmarks, predicted_key_area):
    if image_pil is None or landmarks is None:
        print("Error: Image or Landmarks is None")
        return None

    # image_pil = image_pil.convert("RGB")
    draw = ImageDraw.Draw(image_pil)

    width, height = image_pil.size
    print(f"width: {width}, height: {height}")
    mp_pose = mp.solutions.pose
    pose_connections = mp_pose.POSE_CONNECTIONS

    points = [(int(lm["x"] * width), int(lm["y"] * height)) for lm in landmarks]

    # edge
    for start_idx, end_idx in pose_connections:
        if start_idx < len(points) and end_idx < len(points):
            draw.line([points[start_idx], points[end_idx]], fill=(0, 0, 255), width=3)

    # landmarks
    for idx, (x, y) in enumerate(points):
        radius = 4
        if predicted_key_area[idx] == 1:
            # Blue color for key = 1
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
        else:
            # Gray color for key = 0
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 255))

    return image_pil
 
def predict_key_area(graph):

    # checkpoint = torch.load("rgcn_model.pth", map_location="cpu")
    # print(checkpoint.keys())


    input_dim = 4  # x, y, z, angle
    hidden_dim = 16
    edge_dim = 4  # dir_x, dir_y, dir_z, distance
    model = RGCNNodeModel(input_dim=input_dim, hidden_dim=hidden_dim, edge_dim=edge_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Data Prep
    x = torch.tensor([
        [node["x"], node["y"], node["z"], node.get("angle", 0.0)]
        for _, node in graph.nodes(data=True)
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [u, v] for u, v in graph.edges()
    ], dtype=torch.long).t().contiguous()

    edge_attr = torch.tensor([
        [
            edge_data.get("dir", {}).get("x", 0.0),
            edge_data.get("dir", {}).get("y", 0.0),
            edge_data.get("dir", {}).get("z", 0.0),
            edge_data.get("distance", 0.0)
        ]
        for _, _, edge_data in graph.edges(data=True)
    ], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Predict
    with torch.no_grad():
        prediction = model(data.x, data.edge_index, data.edge_attr)
        pred = (prediction.squeeze(1) > 0).float()

    return pred.tolist()

def draw_predict_key_area_filepath(filepath):
    try:
        image_pil = Image.open(filepath)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    landmarks = get_landmarks(image_pil)
    
    graph = get_graph(landmarks)
    
    predicted_key_area = predict_key_area(graph)
    
    return draw_landmarks(image_pil, landmarks, predicted_key_area)
    
if __name__ == "__main__":
    image_path = "prelim/DATASET1/ให้จี้/bridge.png"
    image = cv2.imread(image_path)
    
    landmarks = get_landmarks(image)
    graph = get_graph(landmarks)
    
    predicted_key_area = predict_key_area(graph)
        
    if predicted_key_area:
        for idx, key in enumerate(predicted_key_area):
            print(f"{idx}: {key}", end=', ')