import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to batch
    predictions = model.predict(input_arr)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Confidence score in percentage
    return class_index, confidence

def main():
    st.title("AI FOOD RECOGNIZE SYSTEM")

    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

    if app_mode == "Home":
        st.header("Introduction")
        st.image("home_img.png", width=500)  # Adjust the width as needed
        
    elif app_mode == "About Project":
        st.subheader("About Dataset")
        st.text("This dataset contains images of the following food items:")
        st.code("Category- ")
        st.code("Cuisines- ")
        st.subheader("Content")
        st.text("This dataset contains three folders:")
        st.text("1. train (100 images each)")
        st.text("2. test (10 images each)")
        st.text("3. validation (10 images each)")

    elif app_mode == "Prediction":
        st.header("Model Prediction")
        test_image = st.file_uploader("Choose an Image:")
        
        if test_image is not None:
            if st.button("Show Image"):
                st.image(test_image, width=400, use_column_width=False)  # Adjust the width as needed

            if st.button("Predict"):
                st.success("Our Prediction")
                class_index, confidence = model_prediction(test_image)
                # Reading Labels
                with open("labels.txt") as f:
                    content = f.readlines()
                label = [i[:-1] for i in content]
                st.success(f"Model predicts it's a {label[class_index]} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    main()
