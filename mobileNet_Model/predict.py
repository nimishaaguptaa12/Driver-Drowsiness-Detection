import cv2
import dlib
import imutils
from scipy.spatial import distance
from time import time

# Load pre-trained models for face and landmarks detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\NIMISHA\Desktop\driver drowsy\dlib_shape_predictor\shape_predictor_68_face_landmarks.dat")

# Define mouth aspect ratio (MAR) threshold for yawning detection
MOUTH_AR_THRESH = 0.3
EYE_AR_THRESH = 0.2

# Function to compute the Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return distance.euclidean(pt1, pt2)

# Function to calculate the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = euclidean_distance(mouth[2], mouth[10])
    B = euclidean_distance(mouth[4], mouth[8])
    C = euclidean_distance(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize webcam feed
cap = cv2.VideoCapture(0)
start_time = time()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)
    for face in faces:
        # Detect landmarks for each face
        landmarks = predictor(gray, face)

        # Get the coordinates for the eyes and mouth
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

        # Calculate the Eye Aspect Ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate the Mouth Aspect Ratio (MAR)
        mar = mouth_aspect_ratio(mouth)

        # Check for drowsiness (if EAR is below threshold)
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for yawning (if MAR is above threshold)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow("Drowsiness and Yawning Detection", frame)

    # Exit condition (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
