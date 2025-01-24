import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.model_selection import GridSearchCV

# Load JSON file
file_path = 'path_to_your_file/svm_datatrain.json'  # แก้ไข path ของไฟล์
with open(file_path, 'r') as f:
    data = json.load(f)

# Extracting angles, distances, and classifications
records = []
for item in data:
    classification = item["classification"]
    # ดึงค่า angle จาก nodes
    angles = [node["angle"] for node in item["nodes"] if "angle" in node]
    # ดึงค่า distance (weight) จาก edges
    distances = [edge["weight"] for edge in item["edges"]]
    
    # รวมค่า angle และ distance เพื่อใช้เป็น feature
    for angle, distance in zip(angles, distances):
        records.append({"angle": angle, "distance": distance, "classification": classification})

# Convert to DataFrame
df = pd.DataFrame(records)

# Features and Target
X = df[['angle', 'distance']]
y = df['classification']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the SVM model
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Train the model with Grid Search
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
grid.fit(X_scaled, y_encoded)

# Get the best model
best_model = grid.best_estimator_
print("Model training complete. Best parameters:", grid.best_params_)

# Save the model, scaler, and label encoder
joblib.dump(best_model, 'svm_leg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("The model, scaler, and label encoder have been saved.")
