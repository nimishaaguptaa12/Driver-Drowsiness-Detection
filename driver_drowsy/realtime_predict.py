import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame  # Import pygame for sound

# Initialize pygame mixer
pygame.mixer.init()

def play_alert():
    try:
        pygame.mixer.music.load(r"C:\Users\NIMISHA\Desktop\driver drowsy\driver_drowsy\alert.wav")  # Update this path to the sound file
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing audio: {e}")

# Constants
EYE_AR_THRESH = 0.2  # Threshold for eye aspect ratio
MOUTH_AR_THRESH = 0.6  # Threshold for mouth aspect ratio
EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames to trigger the alarm
MOUTH_AR_CONSEC_FRAMES = 15  # Number of consecutive frames to trigger the alarm

# Load pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\NIMISHA\Desktop\driver drowsy\dlib_shape_predictor\shape_predictor_68_face_landmarks.dat")  # Update this path if necessary

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Function to detect drowsiness
def detect_drowsiness():
    cap = cv2.VideoCapture(0)

    eye_counter = 0
    mouth_counter = 0
    alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            mar = mouth_aspect_ratio(mouth)
            ear = (ear_left + ear_right) / 2.0

            # Check if drowsiness conditions are met
            if ear < EYE_AR_THRESH and mar > MOUTH_AR_THRESH:
                if not alarm_on:
                    print("ALERT: Drowsiness detected!")
                    play_alert()  # Play the alert sound
                    alarm_on = True
            else:
                alarm_on = False

            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            for (x, y) in mouth:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness()
