import os
import pickle
import config
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import csv
import utils
from datetime import timedelta
import time

# Start timer for tracking runtime
tic = time.time()

# Paths for clustering output and directories
checkpoint_path = "data_checkpoint.pkl"
cluster_path = config.cluster_path
sorted_path = config.sorted_path
csv_path = os.path.join(sorted_path, 'face_clusters.csv')

# Ensure necessary directories exist
utils.check_and_create_dir(cluster_path)
utils.check_and_create_dir(sorted_path)

# Function to safely load images with non-ASCII characters in paths
def load_image(image_path):
    try:
        with open(image_path, 'rb') as file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return image
    except Exception as e:
        print(f"Warning: Could not load image at {image_path}: {e}")
        return None

# Load checkpoint data
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    print(f"Resumed from checkpoint with {len(data)} items in 'data' list.")
else:
    print("No checkpoint found. Exiting.")
    exit()

# Clustering with DBSCAN
encodings = [item["encoding"] for item in data]
dbscan_model = DBSCAN(eps=0.5, min_samples=3, metric="euclidean")
labels = dbscan_model.fit_predict(encodings)
unique_labels = set(labels)

# Prepare CSV to log face details with UTF-8 encoding
with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image Path', 'Location', 'Cluster Label'])

# Process clusters based on DBSCAN results
for label in unique_labels:
    if label == -1:
        # Label -1 indicates noise, or unclustered faces
        continue

    # Create a directory for each unique face cluster
    cluster_dir = os.path.join(sorted_path, f"face_{label}")
    utils.check_and_create_dir(cluster_dir)

    # Collect encodings for the current cluster to save in a .pkl file
    cluster_encodings = []

    for idx, item in enumerate([data[i] for i in range(len(data)) if labels[i] == label]):
        try:
            # Use load_image() to read images with non-ASCII characters
            image = load_image(item["imagePath"])

            # Verify that the image was successfully loaded
            if image is None:
                print(f"Warning: Failed to load image {item['imagePath']}. Skipping this face.")
                continue  # Skip this iteration if the image couldn't be loaded

            # Extract the face region from the image
            top, right, bottom, left = item["loc"]
            face_image = image[top:bottom, left:right]
            cv2.imwrite(os.path.join(cluster_dir, f"face_{label}_{idx}.jpg"), face_image)

            # Add the encoding to the cluster's list of encodings
            cluster_encodings.append(item["encoding"])

            # Append entry to CSV with UTF-8 encoding
            with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([item["imagePath"], item["loc"], label])

        except Exception as e:
            print(f"Error processing face for cluster {label} in file {item['imagePath']}: {e}")

    # Save the cluster encodings to a .pkl file
    pkl_path = os.path.join(cluster_path, f"face_{label}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(cluster_encodings, f)
    print(f"Saved cluster encodings to {pkl_path}")

# Print completion and runtime information
print("Clustering complete. Face data saved to CSV.")
formatted_dur = str(timedelta(seconds=(time.time() - tic)))
print(f'Runtime is {formatted_dur}')
