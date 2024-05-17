import cv2
import dlib
from imutils import face_utils
import numpy as np

# Load the reference image
reference_image = cv2.imread('C:/Users/conor/Python/AiProject/Conor.jpeg')
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/conor/Python/AiProject/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# Detect faces in the reference image
reference_faces = detector(reference_gray, 1)

# Assuming there's only one face in the reference image
if len(reference_faces) == 1:
    reference_landmarks = predictor(reference_gray, reference_faces[0])
    reference_landmarks = face_utils.shape_to_np(reference_landmarks)
    print("Face detected in the reference image.")
else:
    print("Error: More than one face detected in the reference image.")

# Specify the path to save the detected face image
save_path = "C:/Users/conor/Python/AiProject/detected_face.jpg"

# Adjusted similarity threshold (lower value for more leniency)
threshold = 0.05  # Lowered threshold value

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray, 1)
    
    match_found = False  # Flag to track if a match is found
    
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Compare facial landmarks with reference image
        similarity = cv2.matchShapes(np.array(reference_landmarks), np.array(landmarks), cv2.CONTOURS_MATCH_I1, 0)
        print("Similarity:", similarity)
        if similarity < threshold:  # Adjusted threshold
            match_found = True
            break  # Exit the loop if a match is found
    
    # Draw text overlay on the frame based on match status
    if match_found:
        cv2.putText(frame, "Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all  windows
cap.release()
cv2.destroyAllWindows()

