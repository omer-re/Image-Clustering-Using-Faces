import os
import face_recognition
import config
import utils
from face_encoding import get_face_encoding
from face_comparision import compare


def find_cluster_for_new_face(new_face_path):
    """
    Finds the best matching cluster for a new face.
    :param new_face_path: Path to the new face image
    :return: Cluster ID if a match is found, else None
    """
    # Load the new face image and get its encoding
    new_image = face_recognition.load_image_file(new_face_path)
    new_face_encodings = get_face_encoding(new_image, face_recognition)

    if not new_face_encodings:
        print("No face found in the new image.")
        return None

    # Since we assume only one face, take the first encoding
    new_face_encoding = new_face_encodings[0]

    # Load all clusters and compare the new face encoding
    for cluster_file in os.listdir(config.cluster_path):
        cluster_path = os.path.join(config.cluster_path, cluster_file)
        cluster_encodings = utils.load_cluster_in_pickle(cluster_path)

        # Compare the new encoding with encodings in the cluster
        results = compare(cluster_encodings, new_face_encoding, face_recognition)

        # Define match threshold
        if (len(results) > 4 and results.count(True) >= 3) or (len(results) <= 4 and results.count(True) >= 1):
            # If a match is found, return the cluster name (or ID)
            print(f"Match found in cluster: {cluster_file.split('.')[0]}")
            return cluster_file.split('.')[0]

    # No match found
    print("No matching cluster found.")
    return None


# Test with a new face image path
new_face_path = r"path_to_new_face_image.jpg"  # Update with the actual path to the new face image
find_cluster_for_new_face(new_face_path)
