import os
import shutil
import pickle
import config
import face_recognition
import utils
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import csv
from face_loading import loading_face

# Ensure cluster and sorted directories exist
utils.check_and_create_dir(config.cluster_path)
utils.check_and_create_dir(config.sorted_path)
no_face_dir = os.path.join(config.sorted_path, 'no_faces')
utils.check_and_create_dir(no_face_dir)

# Define allowed image extensions
allowed_extensions = {'.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff'}

# Prepare CSV to log face details for easy future matching
csv_path = os.path.join(config.sorted_path, 'face_clusters.csv')
with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image Path', 'Location', 'Cluster Label'])

# Use os.walk to process only image files in the input directory and all subdirectories
all_files = []
for root, dirs, files in os.walk(config.input_path):
    for file in files:
        if os.path.splitext(file.lower())[1] in allowed_extensions:
            all_files.append(os.path.join(root, file))

# Dictionary to track processed faces by unique identifier
processed_faces = {}

# Load images and extract face encodings
data = []
for file_path in tqdm(all_files, total=len(all_files)):
    print(f"Processing file: {file_path}")
    # Check if the file exists before processing
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        continue

    # Load image
    image = loading_face(file_path, face_recognition)
    if image is None:
        continue  # Skip if the image failed to load

    # Detect faces using HOG model for memory efficiency
    face_locations = face_recognition.face_locations(image, model="hog")

    # Check if face locations are found
    if not face_locations:
        print(f"No faces found in {file_path}, moving to no_faces folder.")
        shutil.move(file_path, os.path.join(no_face_dir, os.path.basename(file_path)))
        continue  # Skip this file if no faces are found

    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Process each face location and encoding
    for loc, encoding in zip(face_locations, face_encodings):
        # Create a unique identifier using the image path and face location
        face_id = f"{file_path}_{loc}"

        # Check if this face has already been processed
        if face_id in processed_faces:
            print(f"Skipping duplicate face in {file_path} at location {loc}")
            continue

        # Mark this face as processed
        processed_faces[face_id] = True

        # Append the face encoding and metadata to the data list for clustering
        data.append({"imagePath": file_path, "loc": loc, "encoding": encoding})

# Convert data to numpy array for DBSCAN clustering
encodings = [item["encoding"] for item in data]

# Initialize DBSCAN and fit the model on the encodings
dbscan_model = DBSCAN(eps=0.5, min_samples=3, metric="euclidean")
labels = dbscan_model.fit_predict(encodings)
unique_labels = set(labels)

# Process clusters based on DBSCAN results
for label in unique_labels:
    if label == -1:
        # Label -1 indicates noise, or unclustered faces
        continue

    # Create a directory for each unique face cluster
    cluster_dir = os.path.join(config.sorted_path, f"face_{label}")
    utils.check_and_create_dir(cluster_dir)

    # Collect encodings for the current cluster to save in a .pkl file
    cluster_encodings = []

    for idx, item in enumerate([data[i] for i in range(len(data)) if labels[i] == label]):
        # Load the original image and crop the detected face
        image = cv2.imread(item["imagePath"])
        top, right, bottom, left = item["loc"]
        face_image = image[top:bottom, left:right]
        cv2.imwrite(os.path.join(cluster_dir, f"face_{label}_{idx}.jpg"), face_image)

        # Add the encoding to the cluster's list of encodings
        cluster_encodings.append(item["encoding"])

        # Append entry to CSV with image path, location, and cluster label
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([item["imagePath"], item["loc"], label])

    # Save the cluster encodings to a .pkl file
    pkl_path = os.path.join(config.cluster_path, f"face_{label}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(cluster_encodings, f)
    print(f"Saved cluster encodings to {pkl_path}")

print("Clustering complete. Face data saved to CSV.")

