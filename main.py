import os
import shutil
import config
import face_recognition
from face_loading import loading_face
from face_encoding import get_face_encoding
from face_detection import get_face
from face_comparision import compare
import utils
from tqdm import tqdm

# Ensure cluster and sorted directories exist
utils.check_and_create_dir(config.cluster_path)
utils.check_and_create_dir(config.sorted_path)

# Load existing clusters into memory
existing_clusters = {}
for cluster_file in os.listdir(config.cluster_path):
    cluster_id = cluster_file.split(".")[0]
    cluster_path = os.path.join(config.cluster_path, cluster_file)
    existing_clusters[cluster_id] = utils.load_cluster_in_pickle(cluster_path)

# Start count for any new clusters
count = max([int(cluster_id) for cluster_id in existing_clusters.keys()], default=0) + 1

# Use os.walk to process files in the input directory and all subdirectories
all_files = []
for root, dirs, files in os.walk(config.input_path):
    for file in files:
        all_files.append(os.path.join(root, file))

# Process each file found in all directories and subdirectories
for file_path in tqdm(all_files, total=len(all_files)):
    print(f"Processing file: {file_path}")

    # Load the image
    image = loading_face(file_path, face_recognition)

    # Get all face encodings from the original image
    face_encodings = get_face_encoding(image, face_recognition)
    if face_encodings is None:
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', os.path.basename(file_path)))
        continue

    # Process each detected face encoding in the image
    for face_encoding in face_encodings:
        is_found = False  # Flag to indicate if encoding matched an existing cluster

        # Compare the encoding with each existing cluster
        for cluster_id, encoding_lists in existing_clusters.items():
            results = compare(encoding_lists, face_encoding, face_recognition)
            if (len(results) > 4 and results.count(True) >= 3) or (len(results) <= 4 and results.count(True) >= 1):
                is_found = True
                # Append the encoding to the cluster and save it
                encoding_lists.append(face_encoding)
                utils.save_cluster_in_pickle(os.path.join(config.cluster_path, f"{cluster_id}.pkl"), encoding_lists)
                shutil.copy(file_path, os.path.join(config.sorted_path, cluster_id, os.path.basename(file_path)))
                break

        # If no matching cluster was found, create a new one
        if not is_found:
            new_cluster_id = str(count)
            existing_clusters[new_cluster_id] = [face_encoding]
            utils.create_dir(os.path.join(config.sorted_path, new_cluster_id))
            shutil.copy(file_path, os.path.join(config.sorted_path, new_cluster_id, os.path.basename(file_path)))
            utils.save_cluster_in_pickle(os.path.join(config.cluster_path, f"{new_cluster_id}.pkl"), [face_encoding])
            count += 1
