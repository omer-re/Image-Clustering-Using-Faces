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
from PIL import Image
import numpy as np

# Ensure cluster and sorted directories exist
utils.check_and_create_dir(config.cluster_path)
utils.check_and_create_dir(config.sorted_path)

# Initialize cluster count
cluster_count = sorted(os.listdir(config.cluster_path))
if len(cluster_count) > 0:
    count = int(cluster_count[-1].split('.')[0]) + 1  # Parse last cluster and increment
else:
    count = 0

# Define allowed image extensions
allowed_extensions = {'.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff'}

# Use os.walk to process only image files in the input directory and all subdirectories
all_files = []
for root, dirs, files in os.walk(config.input_path):
    for file in files:
        # Check if the file extension is in the allowed list
        if os.path.splitext(file.lower())[1] in allowed_extensions:
            all_files.append(os.path.join(root, file))

# Helper function to calculate average distance for matching
def is_encoding_match(cluster_encodings, new_encoding, threshold=0.5):
    distances = face_recognition.face_distance(cluster_encodings, new_encoding)
    avg_distance = np.mean(distances)
    return avg_distance < threshold

# Track faces that have already been clustered in each run
clustered_faces = set()

# Process each file found in all directories and subdirectories
for file_path in tqdm(all_files, total=len(all_files)):
    print(f"Processing file: {file_path}")

    # Load the image
    image = loading_face(file_path, face_recognition)

    # Get all face encodings from the original image (not cropped)
    face_encodings = get_face_encoding(image, face_recognition)
    if face_encodings is None:
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', os.path.basename(file_path)))
        continue

    # Process each detected face encoding in the image
    for face_encoding in face_encodings:
        encoding_hash = hash(tuple(face_encoding))  # Create a unique hash for each face encoding
        if encoding_hash in clustered_faces:
            continue  # Skip if this face has already been clustered

        is_found = False  # Flag to indicate if encoding matched an existing cluster

        # Check if there are existing clusters
        if len(os.listdir(config.cluster_path)) > 0:
            for cluster in os.listdir(config.cluster_path):
                # Load encoding list from the cluster
                cluster_path = os.path.join(config.cluster_path, cluster)
                encoding_lists = utils.load_cluster_in_pickle(cluster_path)

                # Use the improved matching function based on average distance
                if is_encoding_match(encoding_lists, face_encoding):
                    is_found = True
                    # Append the encoding to the cluster and save it
                    encoding_lists.append(face_encoding)
                    utils.save_cluster_in_pickle(cluster_path, encoding_lists)
                    shutil.copy(file_path,
                                os.path.join(config.sorted_path, cluster.split(".")[0], os.path.basename(file_path)))
                    clustered_faces.add(encoding_hash)  # Mark face as clustered
                    break

        # If no matching cluster was found, create a new one
        if not is_found:
            new_cluster_id = str(count)
            utils.create_dir(os.path.join(config.sorted_path, new_cluster_id))
            shutil.copy(file_path, os.path.join(config.sorted_path, new_cluster_id, os.path.basename(file_path)))
            utils.save_cluster_in_pickle(os.path.join(config.cluster_path, f"{new_cluster_id}.pkl"), [face_encoding])
            clustered_faces.add(encoding_hash)  # Mark face as clustered
            count += 1

# Thumbnail Summary Generation Function
def generate_cluster_images(thumbnail_size=(100, 100), grid_size=(3, 3)):
    """
    Generate summary images for each cluster using only the face region in thumbnails.
    Each summary image will be named according to its cluster ID.
    :param thumbnail_size: Size of each face thumbnail
    :param grid_size: Layout of thumbnails in the summary image (rows, columns)
    """
    sorted_path = config.sorted_path

    # Iterate through each cluster directory
    for cluster_id in os.listdir(sorted_path):
        cluster_dir = os.path.join(sorted_path, cluster_id)
        if not os.path.isdir(cluster_dir):
            continue  # Skip if not a directory

        # Collect image paths in this cluster
        image_files = [os.path.join(cluster_dir, f) for f in os.listdir(cluster_dir) if f.endswith(('.jpg', '.png'))]

        # Limit to grid size if there are more images than grid slots
        image_files = image_files[:grid_size[0] * grid_size[1]]

        # Create a blank canvas for the grid
        canvas_width = grid_size[1] * thumbnail_size[0]
        canvas_height = grid_size[0] * thumbnail_size[1]
        summary_image = Image.new("RGB", (canvas_width, canvas_height), "white")

        # Place each cropped face thumbnail in the grid
        for index, img_path in enumerate(image_files):
            # Open the image
            image = face_recognition.load_image_file(img_path)

            # Detect face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                continue  # Skip if no face is detected

            # Use the first detected face for thumbnail
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]  # Crop to face region

            # Convert to PIL Image and resize to thumbnail size
            face_pil = Image.fromarray(face_image)
            face_pil.thumbnail(thumbnail_size)

            # Calculate position in the grid
            x_offset = (index % grid_size[1]) * thumbnail_size[0]
            y_offset = (index // grid_size[1]) * thumbnail_size[1]
            summary_image.paste(face_pil, (x_offset, y_offset))

        # Save the summary image with the cluster ID as the filename
        output_path = os.path.join(sorted_path, f"{cluster_id}.jpg")
        summary_image.save(output_path)
        print(f"Saved summary image for cluster {cluster_id} at {output_path}")

# Run the thumbnail generation after clustering
generate_cluster_images()
