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

# Initialize cluster count
cluster_count = sorted(os.listdir(config.cluster_path))
if len(cluster_count) > 0:
    count = int(cluster_count[-1].split('.')[0]) + 1  # Parse last cluster and increment
else:
    count = 0

# Process each file in the input directory
for index, file in tqdm(enumerate(os.listdir(config.input_path)), total=len(os.listdir(config.input_path))):
    file_path = os.path.join(config.input_path, file)
    print(f"Processing file: {file}")  # Debugging statement

    # Load the image
    image = loading_face(file_path, face_recognition)

    # Get all face encodings from the original image (not cropped)
    face_encodings = get_face_encoding(image, face_recognition)
    print(f"Encoding(s) found: {face_encodings is not None}")  # Shows if any encodings were generated

    # If no face encodings are found, save to 'others' category and continue
    if face_encodings is None:
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', file))
        continue

    # Process each detected face encoding in the image
    for face_encoding in face_encodings:
        is_found = False  # Flag to indicate if encoding matched an existing cluster

        # Check if there are existing clusters
        if len(os.listdir(config.cluster_path)) > 0:
            for cluster in os.listdir(config.cluster_path):
                # Load encoding list from the cluster
                cluster_path = os.path.join(config.cluster_path, cluster)
                encoding_lists = utils.load_cluster_in_pickle(cluster_path)

                # Compare the face encoding with the cluster encodings
                results = compare(encoding_lists, face_encoding, face_recognition)

                # Set criteria for matching cluster (based on number of matches)
                if (len(results) > 4 and results.count(True) >= 3) or (len(results) <= 4 and results.count(True) >= 1):
                    is_found = True
                    # Append the encoding to the cluster and save it
                    encoding_lists.append(face_encoding)
                    utils.save_cluster_in_pickle(cluster_path, encoding_lists)
                    shutil.copy(file_path, os.path.join(config.sorted_path, cluster.split(".")[0], file))
                    break

        # If no matching cluster was found, create a new one
        if not is_found:
            utils.create_dir(os.path.join(config.sorted_path, str(count)))
            shutil.copy(file_path, os.path.join(config.sorted_path, str(count), file))
            utils.save_cluster_in_pickle(os.path.join(config.cluster_path, str(count) + ".pkl"), [face_encoding])
            count += 1
