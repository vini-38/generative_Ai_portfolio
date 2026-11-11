    
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️"
)

st.title("CNN Image Classification")
st.write("Upload an image to predict its category.")

# CIFAR-10 class names (10 classes)
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_image_classifier.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=250)

    # Preprocess image
    image = image.resize((32, 32))

    image_array = np.array(image, dtype=np.float32)

    # Normalize image
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)

    # Get predicted class
    predicted_class = int(np.argmax(predictions[0]))

    # Get confidence
    confidence = float(np.max(predictions[0])) * 100

    # Check class index
    if len(predictions[0]) == len(class_names):

        st.success(
            "Prediction: " + class_names[predicted_class]
        )

        st.write(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

    else:
        st.error(
            "Model output does not match CIFAR-10 classes. "
            "Please check your trained model."
        )
