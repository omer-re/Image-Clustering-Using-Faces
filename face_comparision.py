def compare(face_encoding, face_encoding1, face_recognition):
    """
    Compare the face encoding with cluster of encodings
    :param face_encoding: list of encodings
    :param face_encoding1: face encodings
    :param face_recognition: object of face recognition library
    :return: list of results with every encoding from cluster
    """
    results = face_recognition.compare_faces(face_encoding, face_encoding1)
    return results
