############################################
# This file is s
############################################

import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
train_images_1_style = np.load("../Datasets/image_data_X1.npy", allow_pickle=True)
train_images_2_style = np.load("../Datasets/image_data_X2.npy", allow_pickle=True)
train_labels_style = np.load("../Datasets/image_label.npy", allow_pickle=True)

def draw_img(train_images_1, train_images_2, train_labels, image_offset=1000):
    data_size = train_images_1.shape[0]
    if image_offset >= data_size - 4:
        image_offset = data_size - 4
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 레이아웃으로 변경

    for i in range(4):
        ax1 = axes[0, i]  # 첫 번째 행에 첫 번째 이미지
        img1 = train_images_1[image_offset + i].astype('uint8')
        ax1.imshow(img1)
        ax1.set_title(f"Image 1, Label: {train_labels[image_offset + i]}")
        ax1.axis("off")

        ax2 = axes[1, i]  # 두 번째 행에 두 번째 이미지
        img2 = train_images_2[image_offset + i].astype('uint8')
        ax2.imshow(img2)
        ax2.set_title(f"Image 2, Label: {train_labels[image_offset + i]}")
        ax2.axis("off")

    plt.show()
    

unique, counts = np.unique(train_labels_style, return_counts=True)
label_counts = dict(zip(unique, counts))
print("Label Counts:", label_counts)

print(f"Label 0: {label_counts.get(0, 0)}")
print(f"Label 1: {label_counts.get(1, 0)}")

print(train_labels_style)

# draw_img 호출
# draw_img(train_images_1_style, train_images_2_style, train_labels_style, image_offset=400)
# print(train_labels_style)
