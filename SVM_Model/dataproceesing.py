import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm

# Path to dataset
dataset_path = r"C:\Users\NIMISHA\Desktop\driver drowsy\train"

# Load dlib's face detector & landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\NIMISHA\Desktop\driver drowsy\dlib_shape_predictor\shape_predictor_68_face_landmarks.dat")


# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# Function to calculate Mouth Aspect Ratio (MAR) using correct indices
def calculate_mar(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # Vertical distance (upper lip to lower lip)
    B = np.linalg.norm(mouth[4] - mouth[8])  # Another vertical distance
    C = np.linalg.norm(mouth[0] - mouth[6])  # Horizontal width of mouth
    return (A + B) / (2.0 * C)


# Labels for classification
label_mapping = {"Closed": 1, "Open": 0, "yawn": 1, "no_yawn": 0}

# Lists to store features and labels
data, labels = [], []

# Process all images in dataset
for label, class_id in label_mapping.items():
    class_path = os.path.join(dataset_path, label)
    if not os.path.exists(class_path):
        print(f"⚠️ Warning: Folder {class_path} not found. Skipping...")
        continue  # Skip if folder is missing

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {label} images"):
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            mouth = landmarks[48:68]  # Correct mouth landmarks

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            # Save features & labels
            data.append([ear, mar])
            labels.append(class_id)

# Convert and save dataset
data, labels = np.array(data), np.array(labels)
np.save("features.npy", data)
np.save("labels.npy", labels)

print("✅ Dataset processing completed and saved!")
