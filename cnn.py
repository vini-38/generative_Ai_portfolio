import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("CNN Image Classification")
st.write("Upload an image to predict its category")

# Load trained CNN model
model = tf.keras.models.load_model("cnn_image_classifier.keras")

# Get image size from model
image_size = model.input_shape[1:3]

# Change these labels to match your dataset
class_names = ["Class 1", "Class 2"]

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    # Resize image
    image = image.resize((image_size[1], image_size[0]))

    # Convert image into array
    image_array = np.array(image)

    # Normalize image (only if training used /255)
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction[0])

    st.success("Prediction: " + class_names[predicted_class])

    st.write("Confidence:", round(float(np.max(prediction)) * 100, 2), "%")
