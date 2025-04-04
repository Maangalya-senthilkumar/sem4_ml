!pip install opendatasets

import opendatasets as od

od.download("https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits?select=image_labels.csv")




# Install required libraries
!pip install opendatasets kaggle tensorflow keras matplotlib seaborn scikit-learn

# Import libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications import EfficientNetB0  # Faster than MobileNetV2
from keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import mixed_precision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# Enable Mixed Precision Training (Faster on GPU)
mixed_precision.set_global_policy('mixed_float16')

# Dataset paths
train_dir = './handwritten-english-characters-and-digits/augmented_images/augmented_images1'
test_train_dir = './handwritten-english-characters-and-digits/handwritten-english-characters-and-digits/combined_folder/train'
test_test_dir = './handwritten-english-characters-and-digits/handwritten-english-characters-and-digits/combined_folder/test'

# Image parameters
BATCH_SIZE = 128  # Increased batch size for faster training
IMAGE_SIZE = (128, 128)

# Load datasets
train_ds = image_dataset_from_directory(train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True,color_mode='rgb')
test_ds1 = image_dataset_from_directory(test_train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
test_ds2 = image_dataset_from_directory(test_test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Merge test datasets
test_ds = test_ds1.concatenate(test_ds2)

# Normalize image values
train_ds_norm = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds_norm = test_ds.map(lambda x, y: (x / 255.0, y))

# Optimize dataset loading (Caching + Prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds_norm = train_ds_norm.cache().prefetch(buffer_size=AUTOTUNE)
test_ds_norm = test_ds_norm.cache().prefetch(buffer_size=AUTOTUNE)

# Split into training & validation sets (80/20 split)
train_size = int(0.8 * len(train_ds))
val_ds_norm = train_ds_norm.skip(train_size)
train_ds_norm = train_ds_norm.take(train_size)

# Load EfficientNetLite0 (smaller & faster than MobileNetV2)
conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
conv_base.trainable = True  # Freeze base model
for layer in conv_base.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

# Define Model
model = Sequential([
    Input(shape=(128, 128, 3)),
    conv_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(62, activation='softmax')  # 62 classes (digits + uppercase + lowercase)
])

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model (Keeping 50 epochs)
history = model.fit(train_ds_norm, epochs=50, validation_data=val_ds_norm)

# Evaluate Model
print("Training Data Evaluation:")
loss, accuracy = model.evaluate(train_ds_norm)
print("Loss:", loss, "Accuracy:", accuracy)

print("Testing Data Evaluation:")
loss, accuracy = model.evaluate(test_ds_norm)
print("Loss:", loss, "Accuracy:", accuracy)

# Save Model
model.save('optimized_model.keras')
print("Model saved successfully!")

# Load Model
model = keras.models.load_model('optimized_model.keras')
print("Model loaded successfully!")






import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('optimized_model.keras')

# Define class names (digits + uppercase + lowercase letters)
import string
class_names = [str(i) for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)

def preprocess_image(image_path):
    """ Load and preprocess the image for the model. """
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_character(image_path):
    """ Predict the character from the given image. """
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)  # Get highest probability class index
    predicted_char = class_names[predicted_label]  # Map to character
    confidence = np.max(prediction) * 100  # Confidence score

    print(f"Predicted Character: {predicted_char} (Confidence: {confidence:.2f}%)")
    return predicted_char

# Test with an example image
image_path = "/content/a.jpg"  # Change this to your image file
predict_character(image_path)
