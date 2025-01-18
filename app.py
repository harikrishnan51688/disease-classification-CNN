import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
from PIL import Image
import io

MODEL_PATH = 'model_mobilenet.tflite'
LABELS_PATH = 'disease_labels.json'

# Configure page settings
st.set_page_config(
    page_title="Disease Classifier",
    page_icon="🔬",
    layout="centered"
)

# Configure memory growth for GPU if available
@st.cache_resource
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# Load and cache the model
@st.cache_resource
def load_model_and_labels():
    """Load and cache the model and labels"""
    # Load TFLite model
    print("Loading TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully!")

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open(LABELS_PATH, 'r') as f:
        labels_dict = json.load(f)
    labels_map = {v: k for k, v in labels_dict.items()}
    
    return interpreter, labels_map, input_details, output_details

def preprocess_image(img):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    
    return img_array

def format_label(label):
    """Format the prediction label for display"""
    return label.replace('___', ' - ').replace('_', ' ')

def main():
    # Initialize GPU configuration
    configure_gpu()
    
    # Load model and labels
    try:
        interpreter, labels_map, input_details, output_details = load_model_and_labels()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # App title and description
    st.title("Disease Classification System")
    st.write("Upload an image to classify the disease")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_container_width=True)
            
            # Add a prediction button
            if st.button("Analyse"):
                with st.spinner("Processing..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image_display)
                    
                    # Make prediction
                    interpreter.set_tensor(input_details[0]['index'], processed_image)

                    interpreter.invoke()

                    predictions = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Get the prediction results
                    predicted_class = np.argmax(predictions[0])
                    predicted_label = labels_map[predicted_class]
                    confidence = float(predictions[0][predicted_class])
                    
                    # Display results
                    st.success("Prediction Complete!")
                    st.write("### Results:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Predicted Disease:**")
                        st.write(format_label(predicted_label))
                    with col2:
                        st.write("**Confidence:**")
                        st.write(f"{confidence:.2%}")
                    
                    # Display confidence bar
                    st.progress(confidence)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading another image")

if __name__ == "__main__":
    main()