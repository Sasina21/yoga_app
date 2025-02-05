import torch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import pandas as pd
import cv2
from data_preparation import get_graph, get_landmarks

# Load trained model
model_path = "rgcn_model.pth"

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
    
def predict_key_area(graph):

    # checkpoint = torch.load("rgcn_model.pth", map_location="cpu")
    # print(checkpoint.keys())  # ดูว่ามี key อะไรใน state_dict บ้าง


    input_dim = 4  # x, y, z, angle
    hidden_dim = 16
    edge_dim = 4  # dir_x, dir_y, dir_z, distance
    model = RGCNNodeModel(input_dim=input_dim, hidden_dim=hidden_dim, edge_dim=edge_dim)
    model = torch.load("rgcn_model.pt", map_location="cpu")
    model.eval()

    # Data Prep
    # 🔹 ดึงข้อมูลโหนด (nodes)
    x = torch.tensor([
        [node["x"], node["y"], node["z"], node.get("angle", 0.0)]
        for _, node in graph.nodes(data=True)  # ✅ ใช้ graph.nodes(data=True) โดยตรง
    ], dtype=torch.float)

    # 🔹 ดึงข้อมูลขอบ (edges)
    edge_index = torch.tensor([
        [u, v] for u, v in graph.edges()
    ], dtype=torch.long).t().contiguous()  # ✅ ใช้ graph.edges() โดยตรง

    # 🔹 ดึงข้อมูล edge attributes (dir_x, dir_y, dir_z, distance)
    edge_attr = torch.tensor([
        [
            edge_data.get("dir", {}).get("x", 0.0),
            edge_data.get("dir", {}).get("y", 0.0),
            edge_data.get("dir", {}).get("z", 0.0),
            edge_data.get("distance", 0.0)
        ]
        for _, _, edge_data in graph.edges(data=True)  # ✅ ใช้ graph.edges(data=True) โดยตรง
    ], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Predict
    with torch.no_grad():
        prediction = model(data.x, data.edge_index, data.edge_attr)

    return prediction.squeeze().tolist()

if __name__ == "__main__":
    image_path = "prelim/DATASET1/ให้จี้/warrior2_104.jpg"
    image = cv2.imread(image_path)
    
    landmarks = get_landmarks(image)
    graph = get_graph(landmarks)
    
    predicted_key_area = predict_key_area(graph)
        
    if predicted_key_area:
        print(f"Predicted Pose: {predicted_key_area}")