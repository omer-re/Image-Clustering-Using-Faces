import os
import shutil
import config
import face_recognition
from face_loading import loading_face
import utils
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import cv2

# Ensure cluster and sorted directories exist
utils.check_and_create_dir(config.cluster_path)
utils.check_and_create_dir(config.sorted_path)
no_face_dir = os.path.join(config.sorted_path, 'no_faces')
utils.check_and_create_dir(no_face_dir)

# Define allowed image extensions
allowed_extensions = {'.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff'}

# Use os.walk to process only image files in the input directory and all subdirectories
all_files = []
for root, dirs, files in os.walk(config.input_path):
    for file in files:
        if os.path.splitext(file.lower())[1] in allowed_extensions:
            all_files.append(os.path.join(root, file))

# Load images and extract face encodings
data = []
for file_path in tqdm(all_files, total=len(all_files)):
    print(f"Processing file: {file_path}")
    image = loading_face(file_path, face_recognition)

    # Resize the image for memory efficiency during face detection
    small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Scale down to 50%

    # Detect faces using HOG model for memory efficiency
    face_locations = face_recognition.face_locations(small_image, model="hog")

    # Scale face locations back up to original size
    face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]

    # Check if face locations are found
    if not face_locations:
        print(f"No faces found in {file_path}, moving to no_faces folder.")
        shutil.move(file_path, os.path.join(no_face_dir, os.path.basename(file_path)))
        continue  # Skip this file if no faces are found

    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Check if encodings were returned
    if not face_encodings:
        print(f"No valid face encodings for {file_path}, moving to no_faces folder.")
        shutil.move(file_path, os.path.join(no_face_dir, os.path.basename(file_path)))
        continue

    # Store each face location and encoding for clustering
    for loc, encoding in zip(face_locations, face_encodings):
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

    for idx, item in enumerate([data[i] for i in range(len(data)) if labels[i] == label]):
        # Load the original image and crop the detected face
        image = cv2.imread(item["imagePath"])
        top, right, bottom, left = item["loc"]
        face_image = image[top:bottom, left:right]
        cv2.imwrite(os.path.join(cluster_dir, f"face_{label}_{idx}.jpg"), face_image)

print("Clustering complete.")
