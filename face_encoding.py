def get_face_encoding(image, face_recognition):
    """
    Generate face encoding
    :param image: Image matrix
    :param face_recognition: object of face recognition library
    :return:
    """
    face_encoding = face_recognition.face_encodings(image)
    if len(face_encoding) > 0:
        return face_encoding[0]
    else:
        return None
