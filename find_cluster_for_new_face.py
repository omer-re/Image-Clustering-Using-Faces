import os
import pickle
import numpy as np
import face_recognition
import config
from sklearn.metrics.pairwise import cosine_similarity

def find_cluster_for_new_face(new_image_path, threshold=0.5):
    """
    Finds the best matching cluster for a new face encoding from an image.

    Parameters:
    - new_image_path: Path to the new image with the face.
    - threshold: Cosine similarity threshold for determining a match.

    Returns:
    - cluster_label: The cluster label (directory name) if a match is found, None otherwise.
    """
    # Load the new image and extract the face encoding
    image = face_recognition.load_image_file(new_image_path)
    face_locations = face_recognition.face_locations(image, model="hog")
    if not face_locations:
        print("No faces detected in the new image.")
        return None

    # Get the encoding for the first detected face
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if not face_encodings:
        print("No face encodings found in the new image.")
        return None
    new_face_encoding = face_encodings[0]

    # Path to clusters
    cluster_path = config.cluster_path
    best_match_label = None
    highest_similarity = 0

    # Iterate over each .pkl file in the cluster path
    for pkl_file in os.listdir(cluster_path):
        if not pkl_file.endswith('.pkl'):
            continue

        # Load encodings from the .pkl file
        pkl_path = os.path.join(cluster_path, pkl_file)
        with open(pkl_path, 'rb') as f:
            cluster_encodings = pickle.load(f)

        # Calculate similarity with each encoding in the cluster
        similarities = cosine_similarity([new_face_encoding], cluster_encodings)
        max_similarity = np.max(similarities)

        # If similarity exceeds threshold and is the highest similarity so far, update best match
        if max_similarity > threshold and max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match_label = pkl_file.replace('.pkl', '')

    # Report the best matching cluster
    if best_match_label:
        print(f"Best match found in cluster: {best_match_label} with similarity: {highest_similarity}")
    else:
        print("No matching cluster found.")

    return best_match_label

# Example usage
omer_test_new_image_path = r"C:\Users\omerr\Desktop\09_49 31.10.2024(5jn).png"  # Replace with the path to the new face image
omer_test2_new_image_path = r"C:\Users\omerr\Desktop\10_12 31.10.2024(5jp).png"  # Replace with the path to the new face image
shira_test_new_image_path = r"C:\Users\omerr\Desktop\09_50 31.10.2024(5jo).png"  # Replace with the path to the new face image
find_cluster_for_new_face(omer_test_new_image_path)
find_cluster_for_new_face(omer_test2_new_image_path)
find_cluster_for_new_face(shira_test_new_image_path)
