

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the model
MODEL_PATH = r"f:\360 digiTMG\project2\model_saved (1)\model_saved\wood_model.h5"
model = load_model(MODEL_PATH)

# Define a mapping of the labels
week_labels_dict = {
    0: 'WEEK_1',
    1: 'WEEK_2',
    2: 'WEEK_3',
    3: 'WEEK_4',
    4: 'WEEK_5',
    5: 'WEEK_6'
}

# Define the target image size
target_size = (180, 180)

def preprocess_image(image):
    image = np.array(image)
    image_resized = cv2.resize(image, target_size)
    image_scaled = image_resized / 255.0
    return image_scaled

def predict(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return week_labels_dict[predicted_class[0]]

# Streamlit app
st.title('Wood Classification App')
st.write("Upload an image of the wood sample to classify the week.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    label = predict(image)
    st.write(f'The wood sample is from: {label}')



