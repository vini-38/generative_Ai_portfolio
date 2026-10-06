import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_cnn.keras")

# Title
st.title("MNIST Handwritten Digit Classifier")

st.write("Upload a handwritten digit image (0-9)")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    # Display image
    st.image(image, caption="Uploaded Image")

    # Resize image to 28 x 28
    image = image.resize((28, 28))

    # Convert image to numpy array
    image = np.array(image)

    # Normalize pixel values
    image = image / 255.0

    # Reshape image
    image = image.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image)

    # Find predicted digit
    predicted_digit = np.argmax(prediction)

    # Display result
    st.success(f"Predicted Digit: {predicted_digit}")
