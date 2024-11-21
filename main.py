import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt

#Load Model
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
interpreter = tf.lite.Interpreter(model_path="movenet_single_pose_lightning.tflite")
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'Nose → Left Eye',
    (0, 2): 'Nose → Right Eye',
    (1, 3): 'Left Eye → Left Ear',
    (2, 4): 'Right Eye → Right Ear',
    (0, 5): 'Nose → Left Shoulder',
    (0, 6): 'Nose → Right Shoulder',
    (5, 7): 'Left Shoulder → Left Elbow',
    (7, 9): 'Left Elbow → Left Wrist',
    (6, 8): 'Right Shoulder → Right Elbow',
    (8, 10): 'Right Elbow → Right Wrist',
    (5, 11): 'Left Shoulder → Left Hip',
    (6, 12): 'Right Shoulder → Right Hip',
    (11, 13): 'Left Hip → Left Knee',
    (13, 15): 'Left Knee → Left Ankle',
    (12, 14): 'Right Hip → Right Knee',
    (14, 16): 'Right Knee → Right Ankle'
}

#Draw Keypoints
def draw_keypoints(frame, keypoints, confidence):
    y, x, z = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connection(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

#Make Detections
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype = tf.float32)

    #Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_score = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_score)
    
    # Rendering
    draw_connection(frame, keypoints_with_score, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_score, 0.4)
    
    cv2.imshow('MoveNet Lightning', frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
