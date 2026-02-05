pip install opencv-python scikit-learn numpy matplotlib

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_data(data_path, sample_size=2000):
    features = []
    labels = []
    
    # Categories: 0 for Cat, 1 for Dog
    categories = {'cat': 0, 'dog': 1}
    
    # We take a sample to speed up training; SVMs can be slow on 25k images
    count = 0
    for img_name in os.listdir(data_path):
        if count >= sample_size: break
        
        label_str = img_name.split('.')[0]
        label = categories[label_str]
        
        # Read and Resize
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img = cv2.resize(img, (64, 64))
        # Flatten image to a 1D array
        img_flatten = img.flatten() / 255.0  # Normalize pixel values
        
        features.append(img_flatten)
        labels.append(label)
        count += 1
        
    return np.array(features), np.array(labels)

# Path to your 'train' folder from Kaggle
data_path = 'path/to/train' 
X, y = load_data(data_path)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM... grab a coffee, this might take a minute.")
# C is the regularization parameter; gamma defines the influence of single points
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
