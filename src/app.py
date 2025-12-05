from flask import Flask, render_template, request, flash, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2
import os
import tensorflow as tf
import json
import time

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Secret key for flash messages

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the classification labels from a JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels.json'), 'r') as f:
    CLASSIFICATION_LABELS = json.load(f)

# Get the absolute path to the classification model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/classification_model.keras')
# Load the pre-trained classification model
classification_model = load_model(model_path)

def allowed_file(filename):
    """
    Checks if a given filename has an allowed image extension.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Renders the main index page of the web application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads, preprocesses the image, makes a prediction using the
    classification model, and displays the result.
    """
    # Check if a file was part of the request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Process the file if it exists and is allowed
    if file and allowed_file(file.filename):
        # Read the image file into a BytesIO object
        img = Image.open(io.BytesIO(file.read()))
        img_np = np.array(img)
        
        # Convert RGB image to BGR for OpenCV compatibility (if needed for other operations)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Preprocess the image for the classification model (Expected input: 300x300 for EfficientNetV2B3)
        img_resized_classification = cv2.resize(img_np, (300, 300))  # Resize to model's expected input
        img_reshaped_classification = np.reshape(img_resized_classification, (1, 300, 300, 3)) # Reshape for model input
        
        # EfficientNetV2B3 handles normalization internally (expects 0-255 inputs)
        # So we just pass the resized image directly
        img_preprocessed = img_reshaped_classification

        # Run the classification model to get predictions
        prediction = classification_model.predict(img_preprocessed)
        label_index = np.argmax(prediction)  # Get the index of the highest probability class
        label = CLASSIFICATION_LABELS[label_index]  # Get the corresponding label string

        # Generate a unique filename for the output image using a timestamp
        timestamp = str(int(time.time()))
        output_image_filename = f'output_{timestamp}.jpg'
        # Define the path to save the output image in the static folder
        output_image_path = os.path.join('static', output_image_filename)
        # Save the processed image (original BGR version) to the static folder
        cv2.imwrite(output_image_path, img_bgr)

        # Cleanup old images (older than 1 hour)
        cleanup_old_images()

        # Render the result page with the predicted label and image path
        return render_template('result.html', image_path=output_image_filename, label=label, timestamp=timestamp)
    else:
        # Flash an error message for invalid file types and redirect to the index page
        flash('Invalid file type. Please upload an image (png, jpg, jpeg).')
        return redirect(url_for('index'))

def cleanup_old_images(folder='static', age_seconds=3600):
    """
    Removes files in the specified folder that are older than age_seconds.
    """
    try:
        current_time = time.time()
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
        for filename in os.listdir(folder_path):
            if filename.startswith('output_') and filename.endswith('.jpg'):
                file_path = os.path.join(folder_path, filename)
                file_creation_time = os.path.getmtime(file_path)
                if current_time - file_creation_time > age_seconds:
                    os.remove(file_path)
                    print(f"Deleted old image: {filename}")
    except Exception as e:
        print(f"Error cleaning up images: {e}")

if __name__ == '__main__':
    # Get the port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)
