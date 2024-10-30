def loading_face(file_path ,face_recognition):
    """
    Loading image file
    :param file_path: path of a file
    :param face_recognition: object of face recognition library
    :return: image matrix
    """
    image = face_recognition.load_image_file(file_path)

    return image
