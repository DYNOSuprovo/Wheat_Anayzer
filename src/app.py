from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2
import os
import tensorflow as tf
import time

app = Flask(__name__)

# Get absolute model path (safe for Render/Heroku)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/classification_model.h5')

# Load your model
classification_model = load_model(model_path)

# Define labels
CLASSIFICATION_LABELS = ['Crown and Root Rot', 'Healthy Wheat', 'Leaf Rust', 'Wheat Loose Smut']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Preprocess for classification
        img_resized_classification = cv2.resize(img_np, (224, 224))
        img_reshaped_classification = np.reshape(img_resized_classification, (1, 224, 224, 3))
        img_preprocessed = tf.keras.applications.vgg19.preprocess_input(img_reshaped_classification)

        # Predict
        prediction = classification_model.predict(img_preprocessed)
        label_index = np.argmax(prediction)
        label = CLASSIFICATION_LABELS[label_index]

        # Save image with timestamp
        timestamp = str(int(time.time()))
        output_image_filename = f'output_{timestamp}.jpg'
        output_dir = os.path.join('static', 'uploads')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, output_image_filename)
        cv2.imwrite(output_image_path, img_bgr)

        return render_template('result.html', image_path=f'uploads/{output_image_filename}', label=label)

# ------------------- ENTRY POINT -------------------
if __name__ == '__main__':
    # Render/Heroku dynamically assigns PORT
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Starting Flask server on port {port} ...")
    app.run(host='0.0.0.0', port=port)