import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load processed features and labels
data = np.load("features.npy")
labels = np.load("labels.npy")

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize and train SVM classifier
svm_model = SVC(kernel='linear')  # Linear kernel for better interpretability
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Completed! Accuracy: {accuracy:.2%}")

# Save trained model
joblib.dump(svm_model, "svm_drowsiness_model.pkl")
print("✅ Model saved as svm_drowsiness_model.pkl")
