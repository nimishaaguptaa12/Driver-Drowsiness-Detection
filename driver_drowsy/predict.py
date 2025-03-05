import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

MODEL_PATH = "drowsiness_model.h5"

# Load the model
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found! Train the model first using cnn_model.py")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class labels
class_labels = ["Closed", "No Yawn", "Open", "Yawn"]


# Define the preprocessing function to match training preprocessing
def preprocess_image(image_path):
    """Load, resize, and normalize an image for prediction."""
    image_path = image_path.strip().strip('"')

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        exit()

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image. Check file format.")
        exit()

    # Resize and normalize the image (just like during training)
    img = cv2.resize(img, (64, 64))  # Resize to 64x64 as per training
    img = img / 255.0  # Normalize to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return img, cv2.imread(image_path)


def grad_cam(model, img_array, layer_name="conv2d_2"):
    """Generate Grad-CAM heatmap."""
    # Get the output of the last convolutional layer
    grad_model = tf.keras.models.Model(
        inputs=[model.input], outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = np.squeeze(conv_output[0])  # Remove batch dimension
    heatmap = np.dot(heatmap, pooled_grads)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU activation
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay the heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 1 - alpha, jet_heatmap, alpha, 0)
    return superimposed_img


def predict_drowsiness(image_path):
    """Predict drowsiness from an image and visualize Grad-CAM."""
    img, original_img = preprocess_image(image_path)
    prediction = model.predict(img)

    # Print Prediction Confidence Scores
    print("\nPrediction Confidence Scores:")
    for label, score in zip(class_labels, prediction[0]):
        print(f"{label}: {score:.4f}")

    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, img)

    # Overlay heatmap on the original image
    superimposed_img = overlay_heatmap(original_img, heatmap)

    # Display the image with the overlay
    cv2.imshow("Grad-CAM Overlay", superimposed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find the label with the highest score
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    print(f"\nPredicted Drowsiness State: {predicted_label}")


def predict_on_dataset(dataset_path):
    """Predict on a dataset folder with images."""
    # Prepare the data generator with similar preprocessing as training
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Flow from directory (assumes dataset is organized in subdirectories per class)
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # No shuffling, as we want to maintain the order for predictions
    )

    # Predict on the dataset
    print(f"\nPredicting on dataset: {dataset_path}")
    predictions = model.predict(generator, steps=generator.samples // generator.batch_size, verbose=1)

    print("\nPredictions for the dataset:")
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]
        print(f"Image {i + 1}: Predicted Drowsiness State: {predicted_label}")


if __name__ == "__main__":
    # Ask the user if they want to predict a single image or a dataset
    choice = input("Do you want to test a single image or a dataset (single/dataset)? ").strip().lower()

    if choice == "single":
        image_path = input("Enter the full image path: ").strip().strip('"')
        predict_drowsiness(image_path)  # Fixed missing closing parenthesis here
    elif choice == "dataset":
        dataset_path = input("Enter the dataset folder path: ").strip().strip('"')
        predict_on_dataset(dataset_path)
    else:
        print("Invalid choice. Please select 'single' or 'dataset'.")
