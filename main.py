import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import requests

# Define your custom layers with additional parameters
class RandomWidth(tf.keras.layers.Layer):
    def __init__(self, factor=0.2, interpolation='bilinear', seed=None, **kwargs):
        super(RandomWidth, self).__init__(**kwargs)
        self.factor = factor
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=False):
        if training:
            width_factor = tf.random.uniform([], 1 - self.factor, 1 + self.factor, seed=self.seed)
            return tf.image.resize(inputs, [inputs.shape[1], int(inputs.shape[2] * width_factor)], method=self.interpolation)
        return inputs

class RandomHeight(tf.keras.layers.Layer):
    def __init__(self, factor=0.2, interpolation='bilinear', seed=None, **kwargs):
        super(RandomHeight, self).__init__(**kwargs)
        self.factor = factor
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=False):
        if training:
            height_factor = tf.random.uniform([], 1 - self.factor, 1 + self.factor, seed=self.seed)
            return tf.image.resize(inputs, [int(inputs.shape[1] * height_factor), inputs.shape[2]], method=self.interpolation)
        return inputs

# URL of the model file on GitHub or Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1njmx9EULUJW2wB20f4HVHEGQPbvDM_4M'

def main():
    # Set page configuration
    st.set_page_config(page_title='CraveAI', page_icon='🍽️')
    
    # Page title
    # st.title('CraveAI🍽️')
    st.header('AI-driven food identification and taste-based recommendation system ')
    
    # Page title with GIF header
    gif_path = "img/CraveAI.gif"  # Path to your GIF file
    st.image(gif_path, use_column_width=True)

    # About this app
    with st.expander('About this app'):
        st.markdown('**What can this app do?**')
        st.info('This app allows users to recognize food items from images using a pre-trained machine learning model.')

        st.markdown('**How to use the app?**')
        st.warning('Upload an image of a food item, and the app will predict the type of food and provide a confidence score.')

        st.markdown('**Under the hood**')
        st.markdown('Data sets:')
        st.code('''- Train set: 100 images each
    - Test set: 10 images each
    - Validation set: 10 images each
    ''', language='markdown')

        st.markdown('Libraries used:')
        st.code('''- TensorFlow for model prediction
    - NumPy for numerical operations
    - PIL for image processing
    - Streamlit for user interface
    ''', language='markdown')

    # Display training and validation plots
    st.header("Training and Validation Plots")

    # Load and display accuracy plot
    Visual_plot = "img/loss_plot.png"
    # Load and display loss plot
    if os.path.exists(Visual_plot):
        st.image(Visual_plot, caption='Training and Validation Loss')
    
     # Taste Recommender Section
    st.header("Food Recommender Based on Taste")
    st.write("Available food taste: sweet, salty, sour, bitter, spicy")
    user_tastes = st.text_input("What food's tastes do you want? (ex: sweet and sour; salty, sour, and spicy)", key="name")

    def get_taste(x):
        # Convert string of tastes to vector
        tastes = ["sweet", "salty", "sour", "bitter", "spicy"]
        taste_result = []
        
        for taste in tastes:
            if taste in x:
                taste_result.append(1)
            else:
                taste_result.append(0)
        
        return taste_result

    if user_tastes:
        user_taste_vector = get_taste(user_tastes)

        def similarity(x):
            # Calculate similarity of user's taste vector and food's taste vector on database
            taste = np.array(x)
            user_taste = np.array(user_taste_vector)
            
            return np.dot(taste, user_taste)

        data = pd.read_csv("https://raw.githubusercontent.com/JackBboy552/SUTCrave/main/FoodTaste.csv")
        data["taste_vector"] = data["taste"].map(get_taste)
        data["similarity"] = data["taste_vector"].map(similarity)
        filtered_data = data[data["similarity"] > 0]
        filtered_data = filtered_data.sort_values(by = "similarity", ascending = False)
        filtered_data = filtered_data.reset_index(drop = True)
        filtered_data = filtered_data.drop(columns = ["taste_vector", "similarity"])

        st.dataframe(filtered_data)
    else:
        st.warning("Please enter your taste preferences.")
            
    MODEL_PATH = 'trained_model_category.h5'
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as file:
                file.write(response.content)
                
    # Tensorflow Model Prediction
    def model_prediction(test_image):
        # Load the model with custom layers
        with tf.keras.utils.custom_object_scope({'RandomWidth': RandomWidth, 'RandomHeight': RandomHeight}):
            model = tf.keras.models.load_model(MODEL_PATH)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Confidence score in percentage
        return class_index, confidence

    # Prediction Section
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        st.markdown("<h3 style='text-align: left; color: green; font-size: 18px;'>Your Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(test_image, width=400, use_column_width=False)  # Adjust the width as needed

        # if st.button("Show Image"):
        #     st.image(test_image, width=400, use_column_width=False)  # Adjust the width as needed

        if st.button("Predict"):
            progress_text = "Prediction in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            
            class_index, confidence = model_prediction(test_image)
            
            if class_index is not None:
                labels_path = "Category_Labels.txt"
                if os.path.exists(labels_path):
                    with open(labels_path) as f:
                        content = f.readlines()
                    label = [i.strip() for i in content]
                    st.success(f"Category: {label[class_index]}")
                    st.success(f"Accuracy: {confidence:.2f}% ")
                else:
                    st.error("Class index out of range.")
            else:
                st.error("Labels file not found. Please ensure 'Labels.txt' is in the directory.")
                
if __name__ == "__main__":
    main()
