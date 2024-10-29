def get_face_encoding(image, face_recognition):
    """
    Generate face encodings from the original image with face locations detected.
    :param image: The original image
    :param face_recognition: The face recognition library
    :return: List of face encodings, or None if no faces are found
    """
    # Detect face locations in the original image
    face_locations = face_recognition.face_locations(image)

    # Proceed only if there are detected face locations
    if face_locations:
        # Generate face encodings based on detected locations
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
        return face_encodings if face_encodings else None
    return None
