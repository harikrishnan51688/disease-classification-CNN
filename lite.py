import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import tensorflow as tf

# Create Flask app
app = Flask(__name__)

# Global variables
MODEL_PATH = 'model_optimized.tflite'  # Replace with your .tflite model path
LABELS_PATH = 'disease_labels.json'

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels_dict = json.load(f)
labels_map = {v: k for k, v in labels_dict.items()}

# Load TFLite model
print("Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("Model loaded successfully!")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_path):
    """Preprocess the uploaded image for model prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for ResNet101V2
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save the uploaded file temporarily
        temp_path = os.path.join(os.path.dirname(__file__), 'temp_upload.jpg')
        file.save(temp_path)
        
        # Preprocess the image
        processed_image = preprocess_image(temp_path)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Interpret the results
        predicted_class = np.argmax(predictions[0])
        predicted_label = labels_map[predicted_class]
        confidence = float(predictions[0][predicted_class])
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Format prediction label for display
        formatted_label = predicted_label.replace('___', ' - ').replace('_', ' ')
        
        return jsonify({
            'prediction': formatted_label,
            'confidence': f'{confidence:.2%}'
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app
    app.run(debug=False)  # Set debug=False for production
