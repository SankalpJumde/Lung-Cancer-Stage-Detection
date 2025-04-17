import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import json

# Load the trained lung cancer model (ensure the model is saved with '.keras' extension)
model = load_model('models/lung_cancer_model2.keras')

# Parameters
img_height, img_width = 224, 224

# Function to predict lung cancer stage from an image
def predict_stage(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape to match input shape of the model
    predictions = model.predict(img_array)
    stage = np.argmax(predictions[0])
    return stage

# Test the prediction function with an example image
image_path = input('Enter the path of lung cancer image: ')
predicted_stage = predict_stage(image_path)
print(f"Predicted Lung Cancer Stage: {predicted_stage}")