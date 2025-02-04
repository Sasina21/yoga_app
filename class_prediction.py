import os
import joblib
import pandas as pd
import cv2
from data_preparation import extract_graph

# Load trained model, scaler, and label encoder
best_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def predict_pose(image):
    if image is None:
        print(f"Error loading image: class_prediction")
        return None
    
    graph = extract_graph(image)
    
    if not graph:
        print(f"Failed to process graph")
        return None
    
    # Extract features
    angles = []
    dirs = []

    for node_id, node in graph.nodes(data=True):
        if "angle" in node:
            angles.append(node["angle"])

    for _, _, edge_data in graph.edges(data=True):
        if "dir" in edge_data:
            dirs.extend([edge_data["dir"]["x"], edge_data["dir"]["y"], edge_data["dir"]["z"]])
    
    print(f'Extracted {len(angles)} angles and {len(dirs)} direction values')
    
    if not angles or not dirs:
        print(f"Insufficient data extracted")
        return None
    
    # Convert to DataFrame
    feature_data = {
        **{f'angle_{i}': angle for i, angle in enumerate(angles)},
        **{f'dir_{i}': dir_val for i, dir_val in enumerate(dirs)}
    }
    
    feature_df = pd.DataFrame([feature_data])
    
    # Normalize the features using the pre-trained scaler
    print(f"Feature data shape: {feature_df.shape}")
    feature_scaled = scaler.transform(feature_df)
    
    # Predict the class
    prediction = best_model.predict(feature_scaled)
    
    # Decode label
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_class


if __name__ == "__main__":
    image_path = "prelim/DATASET1/ให้จี้/warrior2_104.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
    else:
        predicted_pose = predict_pose(image)
        
        if predicted_pose:
            print(f"Predicted Pose: {predicted_pose}")
