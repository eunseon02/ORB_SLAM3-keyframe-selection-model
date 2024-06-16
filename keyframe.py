#################################
# predict keyframes for ORB-SLAM3
#################################

import cv2
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet import preprocess_input

import matplotlib.pyplot as plt

def load_image(img_path, show=False):
    img = load_img(img_path, target_size=(75, 75))
    img_tensor = img_to_array(img)  # (height, width, 1)
    img_tensor = preprocess_input(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, 1)
    # img_tensor /= 255.0 
    
    return img_tensor

print()
print("------- keyframe selector -------");print()


# load model
model = load_model("./saved_model/best_model")

# image path
img_path = '../Datasets/EuRoC/MH_02_easy/mav0/cam0/data/1403636858651666432.png'
img_dir = '../Datasets/EuRoC/MH_02_easy/mav0/cam0/data/'
img_files = os.listdir(img_dir)

img_files.sort()

# load a single image
prev_keyframe = load_image(img_path)
results = []
# while True:
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    cur_frame = load_image(img_path)
    pred = model.predict([prev_keyframe,cur_frame])
    predicted_labels = (pred > 0.5).astype(int)
    results.append([img_file, predicted_labels[0][0]])
    if predicted_labels == 1:
        prev_keyframe = cur_frame
    print(img_file)
    print(pred)
    print(predicted_labels)



df = pd.DataFrame(results, columns=['Image Path', 'Predicted Label'])

# CSV 파일로 저장
csv_path = './MH02_keyframe_predictions.csv'
df.to_csv(csv_path, index=False)

print(f"Results saved to {csv_path}")