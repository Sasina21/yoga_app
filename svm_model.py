import json
import pandas as pd

# Load the uploaded JSON data
file_path = "output_graphs.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Define relevant node IDs
node_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Prepare data for SVM
data_rows = []
for entry in data:
    filename = entry['filename']
    # Extract relevant nodes
    relevant_nodes = [node for node in entry['nodes'] if node['id'] in node_ids]
    # Create label_set from node.label
    label_set = tuple(node['label'] for node in relevant_nodes)
    # Flatten coordinates (x, y, z) for features
    features = [coord for node in relevant_nodes for coord in (node['x'], node['y'], node['z'])]

