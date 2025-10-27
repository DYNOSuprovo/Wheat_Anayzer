from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2
import os
import tensorflow as tf

import time

app = Flask(__name__)

# Load the models
# detection_model = load_model('models/detection_model.h5')
# Get the absolute path to the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/classification_model.h5')
classification_model = load_model(model_path)

# Define the labels for classification
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

        # Preprocess the image for classification
        img_resized_classification = cv2.resize(img_np, (224, 224))
        img_reshaped_classification = np.reshape(img_resized_classification, (1, 224, 224, 3))
        img_preprocessed = tf.keras.applications.vgg19.preprocess_input(img_reshaped_classification)

        # Run classification model
        prediction = classification_model.predict(img_preprocessed)
        label_index = np.argmax(prediction)
        label = CLASSIFICATION_LABELS[label_index]

        # Save the output image with a unique name
        timestamp = str(int(time.time()))
        output_image_filename = f'output_{timestamp}.jpg'
        output_image_path = os.path.join('static', output_image_filename)
        cv2.imwrite(output_image_path, img_bgr)

        return render_template('result.html', image_path=output_image_filename, label=label, timestamp=timestamp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
