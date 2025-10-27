
import pandas as pd
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# Note: This script assumes that the training images are located in the '../data/detection/train/' directory.
# Please download the images from the Kaggle competition and place them in the correct directory.
# https://www.kaggle.com/c/global-wheat-detection/data

# Load the training data
df_train = pd.read_csv('../data/detection/train.csv')

# Preprocess the data
df_train['bbox'] = df_train['bbox'].apply(lambda x: eval(x))

df_train[['x', 'y', 'w', 'h']] = pd.DataFrame(df_train['bbox'].tolist(), index=df_train.index)

# Prepare the data for training
TRAIN_PATH = '../data/detection/train/'
images = []
labels = []

for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    image_id = row['image_id']
    img = cv2.imread(f'{TRAIN_PATH}{image_id}.jpg', cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append([row['x'], row['y'], row['w'], row['h']])

images = np.array(images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('../src/models/detection_model.h5')
