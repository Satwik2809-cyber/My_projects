import pandas as pd
import numpy as np
import random 
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
import glob
import cv2
from tensorflow.keras.applications.vgg19 import preprocess_input

train_df = pd.read_csv('train.csv')
image_path = 'C:/Users/Sanjana/AppData/landmark/image/0/0/0'

image_paths = ['image_path']
landmark_labels = train_df["landmark_id"]

train_df =train_df.sample(n=min(20000,len(train_df)), random_state=42)

num_classes = len(np.unique(landmark_labels))

test_image_dir = 'C:/Users/Sanjana/AppData/landmark/image/0/0/0'
test_image_path = glob.glob(test_image_dir + '*.jpg')

if len(test_image_path)>20000:
    test_image_path = random.sample(test_image_path, 20000)

test_images = []
for image_path in test_image_path:
    img = Image.open(image_path)
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = preprocess_input(img_array)
    test_images.append(img_array)

test_images = np.array(test_images)

X_train, X_val, y_train, y_val = train_test_split(image_paths,landmark_labels,test_size=0.2,random_state=42)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layers in base_model.layers:
    layers.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())

model.add(Dense(units=num_classes,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


predictions = model.predict(test_images)
print(predictions)