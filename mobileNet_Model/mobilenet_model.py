import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Define constants
IMAGE_SIZE = (224, 224)  # MobileNetV2 requires 224x224 images
BATCH_SIZE = 32
DATASET_PATH = r"C:\Users\NIMISHA\Desktop\driver drowsy\train"

# Data augmentation & preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Load training data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

# Load validation data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

print("Data loaded successfully!")

# Load MobileNetV2 base model (without top layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Freeze base model layers
base_model.trainable = False

# Define the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dense(train_generator.num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("MobileNetV2 model created successfully!")

# Ensure these variables are accessible in train.py
if __name__ == "__main__":
    print("This file should not be executed directly.")
