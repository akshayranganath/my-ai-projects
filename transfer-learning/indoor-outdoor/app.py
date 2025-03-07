import streamlit as st
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# ------------------------------
# Load your trained Keras model
# ------------------------------
@st.cache_resource  # ensures the model is loaded only once
def load_indoor_outdoor_model():
    """
    Load the pre-trained Keras model for indoor/outdoor classification.
    The model is cached to avoid reloading on every run.
    """
    model_path = hf_hub_download(
        repo_id="akshayranganath/indoor-outdoor",  # your HF model repo
        filename="indoor_outdoor_classifier_savedmodel.keras"  # or .h5, whichever you uploaded
    )
    model = load_model(model_path)
    return model

# Load the model once at the start
model = load_indoor_outdoor_model()

# ------------------------------
# Define image preprocessing
# ------------------------------
IMG_SIZE = (224, 224)  # match your training input size

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image to the appropriate format (numpy array) 
    and scales pixel values to [0, 1].
    
    Args:
        img (Image.Image): Input image in PIL format.
    
    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    # Resize to match the training dimensions
    img = img.resize(IMG_SIZE)
    # Convert PIL to numpy
    img_array = image.img_to_array(img)
    # Scale pixels
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_indoor_outdoor(img: Image.Image, threshold=0.5) -> str:
    """
    Given a PIL Image, preprocess and predict using the loaded model.
    Returns 'indoor' or 'outdoor' depending on the predicted probability.
    
    Args:
        img (Image.Image): Input image in PIL format.
        threshold (float): Threshold for classification. Default is 0.5.
    
    Returns:
        str: 'indoor' if predicted probability is below threshold, else 'outdoor'.
    """
    # Preprocess the image
    img_array = preprocess_image(img)
    # Predict using the model
    pred = model.predict(img_array)[0][0]  # single sigmoid output
    # Return the classification result
    return "indoor" if pred < threshold else "outdoor"

# ------------------------------
# Streamlit UI
# ------------------------------
# Set the title of the Streamlit app
st.title("Indoor vs. Outdoor Classifier")

# Provide a brief description of the app
st.write(
    """
    This app classifies an image as either **indoor** or **outdoor**.
    You can either upload an image file or provide an image URL.
    """
)

# Let user pick the input method
input_method = st.radio("Select input method", ("Upload an image", "Image URL"))

if input_method == "Upload an image":
    # Allow user to upload an image file
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Convert uploaded file to PIL image
        pil_image = Image.open(uploaded_file).convert("RGB")
        # Display the uploaded image
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        
        # Predict label
        label = predict_indoor_outdoor(pil_image)
        # Display the prediction result
        st.write(f"**Prediction**: {label}")

elif input_method == "Image URL":
    # Allow user to input an image URL
    image_url = st.text_input("Enter the URL of the image:")
    if st.button("Predict"):
        if image_url:
            try:
                # Fetch the image from the URL
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Convert the fetched content to PIL image
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
                # Display the image from URL
                st.image(pil_image, caption="Image from URL", use_column_width=True)
                
                # Predict label
                label = predict_indoor_outdoor(pil_image)
                # Display the prediction result
                st.write(f"**Prediction**: {label}")
            except Exception as e:
                # Display error if image loading fails
                st.error(f"Error loading image from URL: {e}")
        else:
            # Warn user if URL is empty
            st.warning("Please enter a valid URL.")