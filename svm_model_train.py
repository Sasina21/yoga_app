import json
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

# โหลดข้อมูลจากไฟล์
with open('output_graphs.json', 'r') as f:
    data = json.load(f)

# เตรียมข้อมูล
X = []  # เก็บ input (angle ของ nodes)
y_combined = []  # เก็บ output label ที่รวมกัน

selected_landmarks = [11, 12, 23, 24, 25, 26, 27, 28]

for item in data:
    angles = [node.get('angle', 0) for node in item['nodes']]  # ใช้ค่า angle ถ้ามี ถ้าไม่มีก็ใส่ 0
    X.append(angles)
    
    # สร้าง output label ใหม่
    left = item['left_leg_label']
    right = item['right_leg_label']
    if left == 1 and right == 0:
        y_combined.append("left_leg")
    elif left == 0 and right == 1:
        y_combined.append("right_leg")
    elif left == 1 and right == 1:
        y_combined.append("left_leg+right_leg")
    else:
        y_combined.append("none")  # กรณีที่ไม่มีทั้งสองข้าง

X = np.array(X)
y_combined = np.array(y_combined)

# แปลง output label ให้เป็นตัวเลขสำหรับ SVM
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_combined)

# สเกลข้อมูลเพื่อให้เหมาะสมกับ SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ใช้ Grid Search เพื่อค้นหาค่าพารามิเตอร์ที่ดีที่สุด
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_scaled, y_encoded)

# โมเดลที่ดีที่สุดจาก Grid Search
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# บันทึกโมเดล, scaler และ label encoder ลงไฟล์
joblib.dump(best_model, 'svm_leg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Training completed. Best model, scaler, and label encoder saved.")