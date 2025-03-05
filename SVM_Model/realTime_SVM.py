import cv2
import dlib
import numpy as np
import joblib
from imutils import face_utils

# Load pre-trained SVM model
svm_model = joblib.load("svm_drowsiness_model.pkl")

# Load face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\NIMISHA\Desktop\driver drowsy\dlib_shape_predictor\shape_predictor_68_face_landmarks.dat")


# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # Vertical distance
    B = np.linalg.norm(mouth[4] - mouth[8])  # Another vertical distance
    C = np.linalg.norm(mouth[0] - mouth[6])  # Horizontal width
    return (A + B) / (2.0 * C)


# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:68]

        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        mar = calculate_mar(mouth)

        # Predict drowsiness using the SVM model
        prediction = svm_model.predict([[ear, mar]])[0]

        # Display result
        status = "Awake" if prediction == 0 else "Drowsy"
        color = (0, 255, 0) if status == "Awake" else (0, 0, 255)

        cv2.putText(frame, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
