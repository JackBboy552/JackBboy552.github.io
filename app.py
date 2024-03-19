import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Function to perform image recognition
def recognize_faces(image):
    # Load pre-trained AI model for face detection (e.g., using OpenCV's Haar Cascade classifier)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale
    gray_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(np.array(image), (x, y), (x+w, y+h), (255, 0, 0), 2)

    return np.array(image)

# Streamlit UI
st.title("Image Recognition Web App")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform image recognition on the uploaded image
    recognized_image = recognize_faces(image)

    # Display the image with detected faces
    st.image(recognized_image, caption="Image with Detected Faces", use_column_width=True)
