import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import tempfile

# URLs for the files stored in Google Drive
model_url = "https://drive.google.com/file/d/1ca9nnMI5l0xg50UetwGC3GDwnIIpShpE/view?usp=drive_link"
history_json_url = "https://drive.google.com/file/d/1-66WsU4u_dmhtCh-OYdmzEh8p2vAQvro/view?usp=drive_link"
labels_txt_url = "https://drive.google.com/file/d/1-1lNkU10M8vsq3cgxUbCFSKLxRMDe9_h/view?usp=drive_link"

# Function to download file from Google Drive
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

# Load training history
history_json = download_file(history_json_url).decode("utf-8")
# history = json.loads(history_json)
st.text(f"History JSON content: {history_json[:200]}")  # Print the first 200 characters for debugging

# Check if the history_json is valid JSON
try:
    history = json.loads(history_json)
except json.JSONDecodeError as e:
    st.error(f"Failed to parse JSON: {e}")
    st.stop()

# Load labels
labels_txt = download_file(labels_txt_url).decode("utf-8")
# labels = labels_txt.splitlines()
st.text(f"Labels TXT content: {labels_txt[:200]}")  # Print the first 200 characters for debugging
labels = labels_txt.splitlines()

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
st.header("Model Summary")

with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
    tmp_file.write(download_file(model_url))
    model_path = tmp_file.name

model = tf.keras.models.load_model(model_path)

# Display model summary
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
model_summary = "\n".join(stringlist)
st.text(model_summary)

# Prediction on a single image
st.header("Test Image Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    predicted_class = labels[result_index]

    st.image(uploaded_file, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"Prediction: {predicted_class}")

# Main
if __name__ == "__main__":
    st.write("Streamlit app for model training visualization")
