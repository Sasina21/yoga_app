import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

# โหลดโมเดล, scaler, และ label encoder
svm_model = joblib.load('svm_leg_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# โหลดข้อมูลทดสอบจากไฟล์ JSON
with open('output_graphs_test.json', 'r') as f:
    test_data = json.load(f)

# เตรียมข้อมูลสำหรับการทดสอบ
X_test = []  # เก็บ input (angle ของ nodes)
y_true_combined = []  # เก็บ output label ที่รวมกัน (ground truth)
file_names = []  # เก็บชื่อไฟล์สำหรับอ้างอิง

for item in test_data:
    angles = [node.get('angle', 0) for node in item['nodes']]  # ใช้ค่า angle ถ้ามี ถ้าไม่มีก็ใส่ 0
    X_test.append(angles)
    file_names.append(item['filename'])

    # รวม label จริง (ground truth)
    left = item['left_leg_label']
    right = item['right_leg_label']
    if left == 1 and right == 0:
        y_true_combined.append("left_leg")
    elif left == 0 and right == 1:
        y_true_combined.append("right_leg")
    elif left == 1 and right == 1:
        y_true_combined.append("left_leg+right_leg")
    else:
        y_true_combined.append("none")  # กรณีไม่มี label ใดเลย

X_test = np.array(X_test)

# สเกลข้อมูลใหม่ให้เหมือนกับตอนเทรน
X_test_scaled = scaler.transform(X_test)

# ใช้โมเดลทำการพยากรณ์
predictions = svm_model.predict(X_test_scaled)

# แปลงค่าที่ได้จากตัวเลขกลับไปเป็น label เดิม
predicted_labels = label_encoder.inverse_transform(predictions)

# คำนวณ Accuracy
accuracy = accuracy_score(y_true_combined, predicted_labels)

# แสดงผลลัพธ์
print("Prediction Results:")
for i, label in enumerate(predicted_labels):
    print(f"File: {file_names[i]} → Predicted Label: {label} → Ground Truth: {y_true_combined[i]}")

print(f"\nAccuracy on Test Data: {accuracy * 100:.2f}%")
