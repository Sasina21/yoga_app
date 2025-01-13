import json
import numpy as np
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# เลือกมุมจาก landmark ที่กำหนด
selected_landmarks = [11, 12, 23, 24, 25, 26, 27, 28]

for item in test_data:
    # angles = [node.get('angle', 0) for node in item['node']]
    # # angles = [node.get('angle', 0) for i, node in enumerate(item['nodes']) if i in selected_landmarks]  # ใช้ค่า angle เฉพาะ landmark ที่เลือก

    angles = []
    for node in item['nodes']:
        angle = node.get('angle', 0)
        angles.append(angle)  # ใช้เฉพาะ angle

    X_test.append(angles)
    file_names.append(item['filename'])

    # # รวม label จริง (ground truth)
    # left = item['left_leg_label']
    # right = item['right_leg_label']
    # if left == 1 and right == 0:
    #     y_true_combined.append("left_leg")
    # elif left == 0 and right == 1:
    #     y_true_combined.append("right_leg")
    # elif left == 1 and right == 1:
    #     y_true_combined.append("left_leg+right_leg")
    # else:
    #     y_true_combined.append("none")  # กรณีไม่มี label ใดเลย

    # รวม label จริง (ground truth)
    labels_combined = {
        'left_arm': item.get('left_arm_label', 0),
        'left_upper_arm': item.get('left_upper_arm_label', 0),
        'right_arm': item.get('right_arm_label', 0),
        'right_upper_arm': item.get('right_upper_arm_label', 0),
        'left_leg': item.get('left_leg_label', 0),
        'right_leg': item.get('right_leg_label', 0)
    }
    y_true_combined.append(labels_combined)

X_test = np.array(X_test)

# สเกลข้อมูลใหม่ให้เหมือนกับตอนเทรน
X_test_scaled = scaler.transform(X_test)

# ใช้โมเดลทำการพยากรณ์
predictions = svm_model.predict(X_test_scaled)

# แปลงค่าที่ได้จากตัวเลขกลับไปเป็น label เดิม
predicted_labels = label_encoder.inverse_transform(predictions)

# คำนวณ Accuracy
# accuracy = accuracy_score(y_true_combined, predicted_labels)
accuracy = accuracy_score([str(label) for label in y_true_combined], predicted_labels)

print("Prediction Errors:")
for i, label in enumerate(predicted_labels):
    if label != y_true_combined[i]:
        print(f"File: {file_names[i]} → Predicted Label: {label} → Ground Truth: {y_true_combined[i]}")

print(f"\nAccuracy on Test Data: {accuracy * 100:.2f}%")

# สร้าง DataFrame สำหรับผลลัพธ์
results_df = pd.DataFrame({
    "File": file_names,
    "Predicted Label": predicted_labels,
    "Ground Truth": y_true_combined
})

# บันทึกเป็นไฟล์ CSV
results_df.to_csv('prediction_results.csv', index=False)
print("Results saved to 'prediction_results.csv'")
