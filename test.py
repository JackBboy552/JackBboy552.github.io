import streamlit as st
import tensorflow as tf

def load_model_with_custom_objects():
    try:
        model_path = "trained_model_category.h5"
        with custom_object_scope({'RandomWidth': RandomWidth}):
            model = load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Model Prediction App")

    # Load model
    model_category = load_model_with_custom_objects()
    if model_category is None:
        return

    # Rest of your app logic
    # ...

if __name__ == "__main__":
    main()
