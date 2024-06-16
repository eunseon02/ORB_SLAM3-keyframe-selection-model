
##################################################################
# Convert dataset to .npy file
##################################################################


import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet import preprocess_input
from scipy.spatial.transform import Rotation as R

class DataProcessor:
    def __init__(self, image_dir, data_file, dim=(75, 75)):
        self.image_dir = image_dir
        self.data_file = data_file
        self.dim = dim
        self.data_df = self._load_data()

    def _load_data(self):
        data_df = pd.read_csv(self.data_file, usecols=['im1', 'im2', 'label'])
        return data_df

    def _process_data(self):
        X1 = []
        X2 = []
        y = []

        data_0 = self.data_df[self.data_df['label'] == 0].sample(2000)
        data_1 = self.data_df[self.data_df['label'] == 1].sample(2000)
        sampled_data = pd.concat([data_0, data_1])

        for _, row in sampled_data.iterrows():
            # img_path1 = os.path.join(self.image_dir, f"{int(row['im1']):06}.png")
            # img_path2 = os.path.join(self.image_dir, f"{int(row['im2']):06}.png")
            img_path1 = os.path.join(self.image_dir, row['im1'])
            img_path2 = os.path.join(self.image_dir, row['im2'])

            label = row['label']

            if not os.path.exists(img_path1) or not os.path.exists(img_path2):
                print(f"File not found: {img_path1} or {img_path2}")
                continue

            img1 = load_img(img_path1, target_size=self.dim)
            # img1 = load_img(img_path1)
            img1 = img_to_array(img1)
            img1 = preprocess_input(img1)

            img2 = load_img(img_path2, target_size=self.dim)
            # img2 = load_img(img_path2)
            img2 = img_to_array(img2)
            img2 = preprocess_input(img2)

            X1.append(img1)
            X2.append(img2)
            y.append(label)
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)

        return X1, X2, y


    def save_data(self, save_dir="../Datasets/"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        X1, X2, y = self._process_data()
        
        np.save(os.path.join(save_dir, "image_data_X1_euroc.npy"), X1)
        np.save(os.path.join(save_dir, "image_data_X2_euroc.npy"), X2)
        np.save(os.path.join(save_dir, "image_label_euroc.npy"), y)
        print(f"Data saved to {save_dir}")
        print("Saved files:", os.listdir(save_dir))



# image_dir = '../Datasets/dataset/sequences/00/image_1'
# data_file = '../Datasets/pose_deltas.csv'



image_dir = '../Datasets/EuRoC/MH01/mav0/cam0/data'
data_file = '../Datasets/pose_deltas_euroc.csv'


data_processor = DataProcessor(image_dir, data_file)
data_processor.save_data()
# train_im_1, train_im_2, train_labels = data_processor._process_data()