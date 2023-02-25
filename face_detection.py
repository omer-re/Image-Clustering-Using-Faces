def get_face(image, face_recognition):
    """
    Face Detection
    :param image: Image matrix
    :param face_recognition: object of face recognition library
    :return: Image matrix of face
    """
    locations = face_recognition.face_locations(image)
    if len(locations) > 0:
        # handle only one face and return face with large area
        face_area = 0
        for location in locations:
            y1, x2, y2, x1 = location
            area = (y2 - y1) * (x2 - x1)
            if face_area < area:
                face_area = area
                coord = [y1, x2, y2, x1]
        y1, x2, y2, x1 = coord
        return image[y1:y2, x1:x2]
    else:
        # no face found
        return None
