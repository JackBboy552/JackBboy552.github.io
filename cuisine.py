import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

# Custom layers need to be registered when loading the model
custom_objects = {
    'RandomWidth': tf.keras.layers.RandomWidth,
    'RandomHeight': tf.keras.layers.RandomHeight,
}

# Load the trained models with a progress indicator
with st.spinner('Loading models...'):
    model_category = tf.keras.models.load_model('trained_model_category.h5', custom_objects=custom_objects)
    model_cuisine = tf.keras.models.load_model('trained_model_cuisine.h5', custom_objects=custom_objects)

# Load the label files
def load_labels(labels_path):
    with open(labels_path) as f:
        class_names = [line.strip() for line in f]
    return class_names

category_labels_path = 'Category_Labels.txt'
cuisine_labels_path = 'Cuisine_Labels.txt'
category_labels = load_labels(category_labels_path)
cuisine_labels = load_labels(cuisine_labels_path)

# Function to preprocess and predict individual images
def predict_image(img_path, model, labels):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return labels[predicted_class], confidence

# Streamlit interface
st.title("Image Category and Cuisine Prediction")
st.write("Upload an image to predict its category and cuisine.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save uploaded file to disk
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    img = cv2.imread("temp.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make predictions with a progress indicator
    with st.spinner('Making predictions...'):
        predicted_category, category_confidence = predict_image("temp.jpg", model_category, category_labels)
        predicted_cuisine, cuisine_confidence = predict_image("temp.jpg", model_cuisine, cuisine_labels)
    
    # Display predictions
    st.write(f"**Category:** {predicted_category}")
    st.write(f"**Accuracy:** {category_confidence:.2f}%")
    st.write(f"**Cuisine:** {predicted_cuisine}")
    st.write(f"**Accuracy:** {cuisine_confidence:.2f}%")

    st.success('Prediction complete!')
