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
    st.header('AI-Driven Food Identification and Taste-Based Recommendation System ')
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
    st.header("Model Visualization")

    # Load and display accuracy plot
    loss_plot_path = "img/loss_plot.png"
    cuisine_plot = "img/cuisine_plot.png"
    if os.path.exists(loss_plot_path):
        st.image(loss_plot_path, caption='Single Image Prediction Visualization')
        
    if os.path.exists(cuisine_plot):
        st.image(cuisine_plot, caption='Cuisine Prediction with MobileNetV2')

    # Taste Recommender Section
    st.header("Food Recommender Based on Taste")
    st.write("Available food tastes: sweet, salty, sour, bitter, spicy")
    user_tastes = st.text_input("What food tastes do you want? (e.g., sweet and sour; salty, sour, and spicy)", key="tastes_input")

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
    def model_prediction(test_image):
        # Load the model with custom layers
        model = tf.keras.models.load_model("trained_model.h5")
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Confidence score in percentage
        return class_index, confidence

    # Prediction Section
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", key="category_uploader")

    if test_image is not None:
        image = Image.open(test_image)
        
        # Display the uploaded image
        st.markdown("<h3 style='text-align: left; color: green; font-size: 18px;'>Your Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, width=400, use_column_width=False)

        if st.button("Predict", key="category_predict"):
            progress_text = "Prediction in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            my_bar.empty()

            class_index, confidence = model_prediction(test_image)

            labels_path = "Category_Labels.txt"
            if os.path.exists(labels_path):
                with open(labels_path) as f:
                    content = f.readlines()
                label = [i.strip() for i in content]
                st.success(f"Category: {label[class_index]}")
                st.success(f"Accuracy: {confidence:.2f}% ")
            else:
                st.error("Labels file not found. Please ensure 'Labels.txt' is in the directory.")

    # Cuisine Prediction Section
    st.header("Cuisine Prediction with MobileNetV2")
    st.write("Upload an image of a dish and the model will predict the cuisine type.")

    uploaded_file = st.file_uploader("Choose an image for cuisine prediction...", type=["jpg", "jpeg", "png"], key="cuisine_uploader")

    if uploaded_file is not None:
        cuisine_image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(cuisine_image, caption='Uploaded Image for Cuisine Prediction.', use_column_width=True)
        st.markdown("<h3 style='text-align: left; color: green; font-size: 18px;'>Your Uploaded Image</h3>", unsafe_allow_html=True)

        if st.button("Classify", key="cuisine_predict"):
            with st.spinner('Classifying...'):
                # Rebuild the MobileNetV2 model structure
                base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

                # Add custom layers on top of it
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                predictions = tf.keras.layers.Dense(6, activation='softmax')(x)  # Assuming 6 different cuisine types

                # This is the model we will train
                cuisine_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

                # Load the trained weights
                cuisine_model.load_weights('trained_model_mobilenetv2_cuisine.h5')

                # Define cuisine class labels
                labels_path = 'CuisineLabels.txt'
                with open(labels_path) as f:
                    cuisine_names = [line.strip() for line in f.readlines()]

                # Function to preprocess and predict on a single image
                def predict_single_image(image, model, cuisine_names):
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))  # Resize to match MobileNetV2 input size
                    img = img / 255.0  # Normalize pixel values

                    # Reshape and expand dimensions to match model input
                    img = np.expand_dims(img, axis=0)

                    # Predict
                    predictions = model.predict(img)
                    predicted_cuisine_index = np.argmax(predictions)
                    predicted_cuisine = cuisine_names[predicted_cuisine_index]
                    confidence = np.max(predictions) * 100

                    return predicted_cuisine, confidence

                # Make prediction
                predicted_cuisine, confidence = predict_single_image(cuisine_image, cuisine_model, cuisine_names)

                # Display the prediction
                st.success(f'Cuisine: {predicted_cuisine}')
                st.success(f'Accuracy: {confidence:.2f}%')

if __name__ == "__main__":
    main()
