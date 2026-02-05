import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_gestures(dataset_path):
    images = []
    labels = []
    
    # Traverse through subject folders (00, 01, etc.)
    for subject in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject)
        if not os.path.isdir(subject_path): continue
            
        # Traverse through gesture folders (01_palm, etc.)
        for gesture_folder in os.listdir(subject_path):
            label = int(gesture_folder.split('_')[0]) - 1 # Convert to 0-9
            gesture_path = os.path.join(subject_path, gesture_folder)
            
            for img_file in os.listdir(gesture_path):
                img_array = cv2.imread(os.path.join(gesture_path, img_file), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_array, (64, 64))
                images.append(img_resized)
                labels.append(label)
                
    return np.array(images), np.array(labels)

# Normalize and Reshape
X, y = load_gestures('leapGestRecog/')
X = X.reshape(-1, 64, 64, 1) / 255.0  # Add channel dimension and scale
y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
