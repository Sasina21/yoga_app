import json
import matplotlib.pyplot as plt
import mediapipe as mp

# Path to the JSON file
input_json = "/Users/ngunnnn/Documents/thesis/yoga_app/output_keypoints_with_edges_angles.json"

# เรียกใช้ POSE_CONNECTIONS จาก Mediapipe
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# Function to plot keypoints with Mediapipe skeleton edges in 2D
def plot_keypoints_with_mediapipe_skeleton_2d(json_file, target_filename):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Find the data for the specified filename
    selected_data = next((item for item in data if item["filename"] == target_filename), None)
    
    if not selected_data:
        print(f"Filename {target_filename} not found in the data.")
        return
    
    # Print classification
    classification = selected_data.get("classification")
    print(f"Filename: {target_filename}, Classification: {classification}")
    
    # Extract keypoints (x, y)
    keypoints = []
    for i in range(33):  # Assuming there are 33 keypoints
        x = selected_data.get(f"x_{i}")
        y = selected_data.get(f"y_{i}")
        if x is not None and y is not None:
            keypoints.append((x, y))
    
    # Plotting in 2D
    fig, ax = plt.subplots()
    
    # Plot nodes
    x_vals = [kp[0] for kp in keypoints]
    y_vals = [kp[1] for kp in keypoints]
    ax.scatter(x_vals, y_vals, c='r', label='Keypoints')
    
    # Plot edges using Mediapipe skeleton
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x_line = [keypoints[start_idx][0], keypoints[end_idx][0]]
            y_line = [keypoints[start_idx][1], keypoints[end_idx][1]]
            ax.plot(x_line, y_line, c='b')
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Keypoints and Mediapipe Skeleton (2D) for {target_filename}, classification {classification}")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.show()

# Plot the keypoints with Mediapipe skeleton for the specified filename (2D)
plot_keypoints_with_mediapipe_skeleton_2d(input_json, "warrior162.jpg")
