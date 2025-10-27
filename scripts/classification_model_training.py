import numpy as np
import cv2
import os
from imutils import paths
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Define paths
DATA_DIR = '../data/classification/Images'
MODEL_PATH = '../src/models/classification_model.h5'

# Get the list of labels
LABELS = os.listdir(DATA_DIR)

# Initialize data and labels lists
data = []
labels = []

# Load and preprocess the images
for label in LABELS:
    label_path = os.path.join(DATA_DIR, label)
    for image_path in tqdm(list(paths.list_images(label_path))):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            data.append(image)
            labels.append(label)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Build the model using transfer learning with VGG19
baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(LABELS), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
opt = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Save the model
model.save(MODEL_PATH)
