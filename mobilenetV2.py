# Install required packages
!pip install opendatasets kagglehub tensorflow keras matplotlib seaborn scikit-learn

# Import necessary libraries
import opendatasets as od
import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications import MobileNetV2  # Using a faster model
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import string
import seaborn as sns
import os

%matplotlib inline

# Enable Mixed Precision for Faster Training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from google.colab import files
files.upload()  # This will prompt you to upload kaggle.json


import os
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Set permissions


!pip install kaggle  # Ensure Kaggle API is installed
!kaggle datasets download -d sujaymann/handwritten-english-characters-and-digits
import zipfile

# Unzip the dataset
dataset_zip = "handwritten-english-characters-and-digits.zip"
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Dataset extracted successfully!")


train_dir = './handwritten-english-characters-and-digits/augmented_images/augmented_images1'
test_train_dir = './handwritten-english-characters-and-digits/handwritten-english-characters-and-digits/combined_folder/train'
test_test_dir = './handwritten-english-characters-and-digits/handwritten-english-characters-and-digits/combined_folder/test'


BATCH_SIZE = 64  
IMAGE_SIZE = (128, 128)  

train_ds = image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds1 = image_dataset_from_directory(
    directory=test_train_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_ds2 = image_dataset_from_directory(
    directory=test_test_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = test_ds1.concatenate(test_ds2)

# Normalize image values
train_ds_norm = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds_norm = test_ds.map(lambda x, y: (x / 255.0, y))


AUTOTUNE = tf.data.AUTOTUNE
train_ds_norm = train_ds_norm.cache().prefetch(buffer_size=AUTOTUNE)
test_ds_norm = test_ds_norm.cache().prefetch(buffer_size=AUTOTUNE)

# Split data into train and validation sets
train_size = int(0.8 * len(train_ds))
val_ds_norm = train_ds_norm.skip(train_size)
train_ds_norm = train_ds_norm.take(train_size)


# MobileNetV2 Model for Faster Training
conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
conv_base.trainable = False  # Freeze base model

model = Sequential([
    Input(shape=(128, 128, 3)),
    conv_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(62, activation='softmax')  # 10 digits + 26 uppercase + 26 lowercase = 62 classes
])

# Compile the model with Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds_norm, epochs=50, validation_data=val_ds_norm)  # Reduced epochs to 20


# Evaluate model
print("Training Data Evaluation:")
loss, accuracy = model.evaluate(train_ds_norm)
print("Loss:", loss, "Accuracy:", accuracy)

print("Testing Data Evaluation:")
loss, accuracy = model.evaluate(test_ds_norm)
print("Accuracy:", accuracy)




# Generate confusion matrix
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true, y_pred = np.array(y_true), np.array(y_pred)
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
class_names = [str(i) for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(20, 20))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, square=True, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Save model
model.save('optimized_model.keras')


from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('optimized_model.keras')
print("Model loaded successfully!")


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(image_path):
    # Load the image and resize it to (128, 128) instead of 224x224
    img = image.load_img(image_path, target_size=(128, 128))
    
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    
    # Normalize the image (scale pixel values to [0,1])
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input shape (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_character(image_path):
    # Preprocess the input image
    img_array = preprocess_image(image_path)
    
    # Get model predictions
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Class names (Digits + Uppercase + Lowercase letters)
    class_names = [str(i) for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    
    print(f"Predicted Character: {class_names[predicted_class]}")

# Example Usage
predict_character('/content/42fa6489-c4b8-48c5-a959-14ec09ad0c05.jpg')


def predict_character(image_path):
    # Preprocess the input image
    img_array = preprocess_image(image_path)
    
    # Get model predictions
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Class names (Digits + Uppercase + Lowercase letters)
    class_names = [str(i) for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    
    print(f"Predicted Character: {class_names[predicted_class]}")

# Example Usage
predict_character('/content/a82ce265-4d16-4fba-be6b-7147d61cf685.jpg')
