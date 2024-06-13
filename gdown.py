import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import gdown
import tempfile
import time
import os
import requests
import h5py

# Function to download file from a Google Drive URL using gdown
def download_file_from_google_drive(url, filename):
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = os.path.join(tempfile.gettempdir(), filename)
    gdown.download(download_url, output, quiet=False)
    return output

# Function to load JSON content
def load_json_from_url(url):
    json_file = download_file_from_google_drive(url, 'history.json')
    with open(json_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            st.stop()

# URLs for the files stored in Google Drive
model_url = "https://drive.google.com/file/d/1ca9nnMI5l0xg50UetwGC3GDwnIIpShpE/view?usp=drive_link"
history_json_url = "https://drive.google.com/file/d/1-66WsU4u_dmhtCh-OYdmzEh8p2vAQvro/view?usp=drive_link"
labels_txt_url = "https://drive.google.com/file/d/1-1lNkU10M8vsq3cgxUbCFSKLxRMDe9_h/view?usp=drive_link"

# Load training history from JSON
history = load_json_from_url(history_json_url)

# Load labels
labels_txt = download_file_from_google_drive(labels_txt_url, 'labels.txt')
with open(labels_txt, 'r') as f:
    labels = f.read().splitlines()

# Streamlit App
st.title("Model Training and Evaluation Results")

# Plot accuracy and loss
st.header("Training and Validation Accuracy/Loss")
epochs = range(1, len(history['accuracy']) + 1)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(epochs, history['accuracy'], 'r', label='Training accuracy')
ax[0].plot(epochs, history['val_accuracy'], 'b', label='Validation accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(epochs, history['loss'], 'r', label='Training loss')
ax[1].plot(epochs, history['val_loss'], 'b', label='Validation loss')
ax[1].set_title('Training and Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()

st.pyplot(fig)

# Download and load the model
model_file = download_file_from_google_drive(model_url, 'model.h5')

# Verify that the downloaded file is a valid HDF5 file
try:
    with h5py.File(model_file, 'r') as f:
        pass
except OSError as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

model = tf.keras.models.load_model(model_file)

# Function to perform prediction
def model_prediction(image_file):
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return result_index, confidence

# Prediction on a single image
st.header("Test Image Prediction")
test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if test_image is not None:
    st.markdown("<h3 style='text-align: left; color: green; font-size: 18px;'>Your Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(test_image, width=400, use_column_width=False)

    if st.button("Predict"):
        progress_text = "Prediction in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        class_index, confidence = model_prediction(test_image)
        st.success(f"Model predicts it's a {labels[class_index]} with {confidence:.2f}% confidence.")
