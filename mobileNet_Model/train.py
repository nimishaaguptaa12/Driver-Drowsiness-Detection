import os
import tensorflow as tf
from mobilenet_model import model, train_generator, val_generator

# Define constants
EPOCHS = 10  # You can increase this for better accuracy
MODEL_SAVE_PATH = "models/mobilenet_drowsiness.h5"

# Ensure the models/ directory exists
os.makedirs("models", exist_ok=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f" Model trained and saved to {MODEL_SAVE_PATH}")

# Display final accuracy
final_acc = history.history['val_accuracy'][-1] * 100
print(f" Final Validation Accuracy: {final_acc:.2f}%")
