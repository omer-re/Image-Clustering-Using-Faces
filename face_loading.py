from PIL import Image
import numpy as np
import face_recognition


def loading_face(file_path, face_recognition_module):
    """
    Loads an image from file_path and converts it to RGB.

    Parameters:
    - file_path: Path to the image file.
    - face_recognition_module: Reference to face_recognition module to avoid direct import issues.

    Returns:
    - Numpy array of the image in RGB format.
    """
    try:
        image = face_recognition_module.load_image_file(file_path)
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None
