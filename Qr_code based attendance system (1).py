#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import os
import numpy as np
import cv2
import openpyxl
from pyzbar.pyzbar import decode
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import subprocess


# # Loading and Preprocessing Data

# In[2]:


# Function to load and preprocess images
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (100, 100))  # Resize image to desired dimensions
        if img is not None:  # Check if the image is valid
            images.append(img)
            label = filename.split('_')[0]  # Extract label from filename
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess dataset
dataset_folder = 'qr_dataset'  # Path to your dataset folder
images, labels = load_images(dataset_folder)

if len(images) > 0:
    print("Dataset loaded successfully.")
    print("Number of images:", len(images))
else:
    print("Failed to load the dataset. Please check the dataset folder path.")


# In[3]:


# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)


# # Dataset Splitting

# In[4]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Convert labels to categorical
num_classes = len(np.unique(labels_encoded))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# # Build CNN for Feature Extraction 

# In[5]:


# Build CNN model for feature extraction
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# # Normalize pixel values to be between 0 and 1

# In[6]:


# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# In[7]:


# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 
# # Training the CNN Model

# In[8]:


# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# # Save the CNN model

# In[9]:


# Save the CNN model
cnn_model.save('cnn_model.h5')


# # Use CNN for feature extraction

# In[10]:


# Use CNN for feature extraction
cnn_features_train = cnn_model.predict(X_train)
cnn_features_test = cnn_model.predict(X_test)


# # Train and evaluate SVM model

# In[11]:


# Train and evaluate SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(cnn_features_train, np.argmax(y_train, axis=1))

# Evaluate and save SVM model
svm_pred = svm_model.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), svm_pred)
print(f'Test accuracy for SVM:', accuracy)
print(classification_report(np.argmax(y_test, axis=1), svm_pred))
joblib.dump(svm_model, 'svm_model.pkl')


# # Train and Evaluate KNN Model
# 

# In[12]:


# Train and evaluate KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(cnn_features_train, np.argmax(y_train, axis=1))

# Evaluate and save KNN model
knn_pred = knn_model.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), knn_pred)
print(f'Test accuracy for KNN:', accuracy)
print(classification_report(np.argmax(y_test, axis=1), knn_pred))
joblib.dump(knn_model, 'knn_model.pkl')


# # Train and Evaluate Random Forest Model

# In[13]:


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(cnn_features_train, np.argmax(y_train, axis=1))

# Evaluate and save Random Forest model
rf_pred = rf_model.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), rf_pred)
print(f'Test accuracy for Random Forest:', accuracy)
joblib.dump(rf_model, 'rf_model.pkl')


# # Train and Evaluate Logistic Regression Model

# In[14]:


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(cnn_features_train, np.argmax(y_train, axis=1))

# Evaluate and save Logistic Regression model
lr_pred = lr_model.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), lr_pred)
print(f'Test accuracy for Logistic Regression:', accuracy)
joblib.dump(lr_model, 'lr_model.pkl')


# # Evaluate the model

# In[15]:


# Evaluate the model
accuracy = accuracy_score(np.argmax(y_test, axis=1), svm_pred)
print('Test accuracy:', accuracy)


# # Real-time QR Code Attendance Tracking System Using Python

# In[16]:


# Function to update Excel file with QR code data
def update_excel(qr_data):
    try:
        workbook = openpyxl.load_workbook('qr_code_attendance.xlsx')
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([qr_data])  # Append QR code data to the Excel sheet
    workbook.save('qr_code_attendance.xlsx')

# Initialize variables to store the last scanned QR code data
last_scanned_qr_code = None

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Decode QR code
    decoded_objects = decode(frame)

    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')
        
        # Check if QR code is different from the last scanned QR code
        if qr_data != last_scanned_qr_code:
            update_excel(qr_data)
            print(f"QR Code Data: {qr_data}")
            last_scanned_qr_code = qr_data
    
    cv2.imshow('QR Code Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




