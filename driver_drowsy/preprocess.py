from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
DATASET_PATH = "C:/Users/NIMISHA/Desktop/driver drowsy/train"

# Data Augmentation and Rescaling
datagen = ImageDataGenerator(
    rescale=1./255,  # Rescaling to normalize pixel values to the range [0, 1]
    rotation_range=30,  # Random rotations within a range of 30 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Random shearing transformations
    zoom_range=0.2,  # Random zooming
    horizontal_flip=True,  # Random horizontal flipping
    fill_mode='nearest',  # Fill pixels after transformations with nearest value
    validation_split=0.2  # Keep 20% data for validation
)

# Train data generator
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Set as training data subset
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Set as validation data subset
)

# Print status of dataset processing
train_batches = train_generator.samples // BATCH_SIZE
val_batches = val_generator.samples // BATCH_SIZE

print(f"Training dataset processed: {train_batches} batches")
print(f"Validation dataset processed: {val_batches} batches")

# Model definition (example CNN model)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_batches,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_batches,
    callbacks=[early_stopping]
)

# Print accuracy of the model
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
