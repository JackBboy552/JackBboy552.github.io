import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def main():
    st.set_page_config(page_title='CraveAI', page_icon='🍽️')
    st.header('AI-driven food identification and taste-based recommendation system ')
    
    # Test Image Upload
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, width=400, use_column_width=False)  # Display uploaded image
        if st.button("Predict"):
            with st.spinner("Loading model and making prediction..."):
                try:
                    # Register custom layers
                    custom_objects = {
                        'RandomFlip': tf.keras.layers.RandomFlip,
                        'RandomRotation': tf.keras.layers.RandomRotation,
                        'RandomZoom': tf.keras.layers.RandomZoom,
                        'RandomWidth': tf.keras.layers.RandomWidth,
                        'RandomHeight': tf.keras.layers.RandomHeight,
                    }
                    tf.keras.utils.get_custom_objects().update(custom_objects)
                    
                    # Load the model with custom layers
                    model = tf.keras.models.load_model("trained_model_vgg16.h5", custom_objects=custom_objects)
                    
                    image = Image.open(test_image)
                    image = image.resize((64, 64))
                    input_arr = np.array(image)
                    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch

                    predictions = model.predict(input_arr)
                    class_index = np.argmax(predictions)
                    confidence = np.max(predictions) * 100  # Confidence score in percentage

                    # Load labels
                    labels_path = "Labels.txt"
                    if os.path.exists(labels_path):
                        with open(labels_path) as f:
                            content = f.readlines()
                        label = [i.strip() for i in content]
                        st.success(f"Category: {label[class_index]}")
                        st.success(f"Accuracy: {confidence:.2f}%")
                    else:
                        st.error("Labels file not found. Please ensure 'Labels.txt' is in the directory.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
