import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256  # Image input size for the model (adjust if needed)

# Load the model
@st.cache_resource
def load_model_file():
    try:
        model = load_model('densenet201.h5')  # Replace with your model file
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model_file()

# Function to get class labels (if any)
def get_class_labels(model):
    try:
        if hasattr(model, 'class_names'):
            return model.class_names
        else:
            return [f"Class {i}" for i in range(model.output_shape[-1])]
    except Exception as e:
        st.error(f"Error retrieving class labels: {e}")
        return None

class_labels = get_class_labels(model)

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image to classify it using the pre-trained model.")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        st.write("Processing the image...")
        img = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to match model's expected input size
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction
        if st.button("Show Prediction"):
            if model is not None:
                prediction = model.predict(img_array)
                predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Get the index of the top class
                confidence = np.max(prediction) * 100

                # Display results
                st.write("Prediction Results:")
                if class_labels:
                    st.write(f"Class: {class_labels[predicted_class_idx]}")
                else:
                    st.write(f"Class Index: {predicted_class_idx}")
                st.write(f"Confidence: {confidence:.2f}%")

                # Optional: Show all class probabilities
                if st.checkbox("Show All Class Probabilities"):
                    probabilities = {class_labels[i] if class_labels else f"Class {i}": prob
                                     for i, prob in enumerate(prediction[0])}
                    st.write(probabilities)
            else:
                st.error("Model not loaded. Please check the model file.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.write("Please upload an image to proceed.")
