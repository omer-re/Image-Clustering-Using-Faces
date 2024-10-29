import face_recognition

# Path to one of the images causing issues
image_path = r"G:\My Drive\OMER_PERSONAL\Omer\Gallery\Wedding\magnets\test\_DSC7186.jpg"

# Load the image
image = face_recognition.load_image_file(image_path)

# Detect face locations
face_locations = face_recognition.face_locations(image)
print(f"Face locations: {face_locations}")

# If face locations are found, proceed to generate face encodings
if face_locations:
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    print(f"Face encodings: {face_encodings}")
else:
    print("No face locations found.")
