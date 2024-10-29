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

# Ensure cluster and sorted directories exist
utils.check_and_create_dir(config.cluster_path)
utils.check_and_create_dir(config.sorted_path)

# Initialize cluster count
cluster_count = sorted(os.listdir(config.cluster_path))
if len(cluster_count) > 0:
    count = int(cluster_count[-1].split('.')[0]) + 1  # Parse last cluster and increment
else:
    count = 0

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

    # Get all face encodings from the original image (not cropped)
    face_encodings = get_face_encoding(image, face_recognition)
    if face_encodings is None:
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', os.path.basename(file_path)))
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
                    shutil.copy(file_path,
                                os.path.join(config.sorted_path, cluster.split(".")[0], os.path.basename(file_path)))
                    break

        # If no matching cluster was found, create a new one
        if not is_found:
            new_cluster_id = str(count)
            utils.create_dir(os.path.join(config.sorted_path, new_cluster_id))
            shutil.copy(file_path, os.path.join(config.sorted_path, new_cluster_id, os.path.basename(file_path)))
            utils.save_cluster_in_pickle(os.path.join(config.cluster_path, f"{new_cluster_id}.pkl"), [face_encoding])
            count += 1


# Thumbnail Summary Generation Function
def generate_cluster_images(thumbnail_size=(100, 100), grid_size=(5, 5)):
    """
    Generate summary images for each cluster with thumbnails of faces.
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

        # Place each image as a thumbnail in the grid
        for index, img_path in enumerate(image_files):
            # Open the image and create a thumbnail
            with Image.open(img_path) as img:
                img.thumbnail(thumbnail_size)
                x_offset = (index % grid_size[1]) * thumbnail_size[0]
                y_offset = (index // grid_size[1]) * thumbnail_size[1]
                summary_image.paste(img, (x_offset, y_offset))

        # Save the summary image with the cluster ID as the filename
        output_path = os.path.join(sorted_path, f"{cluster_id}.jpg")
        summary_image.save(output_path)
        print(f"Saved summary image for cluster {cluster_id} at {output_path}")


# Run the thumbnail generation after clustering
generate_cluster_images()
