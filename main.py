import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time
import os

def main():
    # Set page configuration
    st.set_page_config(page_title='CraveAI', page_icon='🍽️')

    # Page title with GIF header
    st.header('AI-driven food identification and taste-based recommendation system ')
    gif_path = "img/CraveAI.gif"  # Path to your GIF file
    if os.path.exists(gif_path):
        st.image(gif_path, use_column_width=True)

    # About this app
    with st.expander('About this app'):
        st.markdown('**What can this app do?**')
        st.info('This app provides a comprehensive AI-driven food identification and taste-based recommendation system.\n\n'
                'The model used in this app is based on MobileNetV2, fine-tuned on the provided datasets for both category and cuisine classification.')

        st.markdown('**How to use the app?**')
        st.warning('1. **Prediction Section**: Upload an image, and the app will predict the category of the food item with a confidence score. The app also displays the uploaded image and the predicted category with accuracy.\n'
                   '2. **Taste Recommender**: The taste recommender feature allows users to input their preferred tastes (sweet, salty, sour, bitter, spicy) and provides food recommendations based on a pre-defined taste profile dataset.')

        st.markdown('**Under the hood**')
        st.markdown('Datasets:')
        st.code('''- Category (Food, Dessert, Pastry, Drink)
    - Train set: 160 images each
    - Test set: 40 images each
    - Validation set: 40 images each
    ''', language='markdown')
        st.code('''- Cuisine (Chinese, Malay, Japanese, Philippino, Indian, Thai)
    - Train set: 400 images each
    - Test set: 100 images each
    - Validation set: 100 images each
    ''', language='markdown')

        st.markdown('Libraries used:')
        st.code('''- TensorFlow for model prediction
    - NumPy for numerical operations
    - Pandas for data manipulation
    - Streamlit for user interface
    - OpenCV for image processing
    - Matplotlib for plotting
    ''', language='markdown')

    # Display training and validation plots
    st.header("Training and Validation Plots")

    # Load and display accuracy plot
    loss_plot_path = "img/loss_plot.png"
    if os.path.exists(loss_plot_path):
        st.image(loss_plot_path, caption='Training and Validation Loss')

    # Taste Recommender Section
    st.header("Food Recommender Based on Taste")
    st.write("Available food tastes: sweet, salty, sour, bitter, spicy")
    user_tastes = st.text_input("What food tastes do you want? (e.g., sweet and sour; salty, sour, and spicy)", key="tastes")

    def get_taste_vector(taste_input):
        # Convert string of tastes to vector
        tastes = ["sweet", "salty", "sour", "bitter", "spicy"]
        taste_vector = [1 if taste in taste_input else 0 for taste in tastes]
        return taste_vector

    if user_tastes:
        user_taste_vector = get_taste_vector(user_tastes)

        def calculate_similarity(taste_vector):
            # Calculate similarity between user's taste vector and food's taste vector in the database
            taste_array = np.array(taste_vector)
            user_taste_array = np.array(user_taste_vector)
            return np.dot(taste_array, user_taste_array)

        data = pd.read_csv("https://raw.githubusercontent.com/JackBboy552/SUTCrave/main/FoodTaste.csv")
        data["taste_vector"] = data["taste"].apply(get_taste_vector)
        data["similarity"] = data["taste_vector"].apply(calculate_similarity)
        filtered_data = data[data["similarity"] > 0].sort_values(by="similarity", ascending=False).reset_index(drop=True)
        filtered_data = filtered_data.drop(columns=["taste_vector", "similarity"])

        st.dataframe(filtered_data)
    else:
        st.warning("Please enter your taste preferences.")

    # Tensorflow Model Prediction
    def model_prediction(test_image, model_path, input_size, num_classes, labels_path):
        # Rebuild the MobileNetV2 model structure
        base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=input_size)

        # Add custom layers on top of it
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Number of classes

        # This is the model we will train
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

        # Load the trained weights
        model.load_weights(model_path)

        image = tf.keras.preprocessing.image.load_img(test_image, target_size=input_size[:2])
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Confidence score in percentage

        with open(labels_path) as f:
            content = f.readlines()
        labels = [i.strip() for i in content]

        return labels[class_index], confidence

    # Prediction Section
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        image = Image.open(test_image)
        
        # Display the uploaded image
        st.markdown("<h3 style='text-align: left; color: green; font-size: 18px;'>Your Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, width=400, use_column_width=False)

        if st.button("Predict"):
            progress_text = "Prediction in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            my_bar.empty()

            category_label, category_confidence = model_prediction(test_image, 'trained_model_mobilenetv2.h5', (64, 64, 3), 4, 'Category_Labels.txt')
            cuisine_label, cuisine_confidence = model_prediction(test_image, 'trained_model_mobilenetv2_cuisine.h5', (224, 224, 3), 6, 'CuisineLabels.txt')

            st.success(f"Category: {category_label}")
            st.success(f"Accuracy: {category_confidence:.2f}%")
            st.success(f"Cuisine: {cuisine_label}")
            st.success(f"Accuracy: {cuisine_confidence:.2f}%")


if __name__ == "__main__":
    main()
