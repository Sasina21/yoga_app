import os
import json
import glob
from class_prediction import predict_pose_filepath

IMAGE_FOLDER = "Images/References"
OUTPUT_FILE = "storage.json"

def process_images():
    results = {}

    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.*"))
    
    for img_path in image_paths:
        label = predict_pose_filepath(img_path)
        results[img_path] = label

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    process_images()
