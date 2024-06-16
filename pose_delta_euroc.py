##################################################################
# Calculate the change in pose between frames in the EuRoC dataset
##################################################################

import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet import preprocess_input
from scipy.spatial.transform import Rotation as R

class EuRoCDataset:
    def __init__(self, image_dir, pose_file, dim=(75, 75), output_dir="../Datasets/"):
        self.image_dir = image_dir
        self.pose_file = pose_file
        self.dim = dim
        self.output_dir = output_dir
        self.pose_df = self._load_poses()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _load_poses(self):
        pose_df = pd.read_csv(self.pose_file)
        pose_df.columns = ['timestamp', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']
        pose_df = pose_df[['timestamp', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']]
        pose_df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']
        return pose_df

    def compute_deltas(self):
        deltas = []
        X1 = []
        X2 = []
        y = []
        frame_intervals = [1, 2, 3, 4, 5]  # Define the frame intervals you want to use
        num_poses = len(self.pose_df)
        print(num_poses)

        for i in range(num_poses):
            row1 = self.pose_df.iloc[i]
            pos1 = np.array([row1['tx'], row1['ty'], row1['tz']])
            quat1 = np.array([row1['qx'], row1['qy'], row1['qz'], row1['qw']])
            # file_path1 = str(i).zfill(6)
            # img_path1 = os.path.join(self.image_dir, f"{file_path1}.png")
            timestamp1 = str(int(float(row1['timestamp'])))
            img_path1 = os.path.join(self.image_dir, f"{timestamp1}.png")


            if not os.path.exists(img_path1):
                print(f"File not found: {img_path1}")
                continue

            img1 = load_img(img_path1, target_size=self.dim)
            img1 = img_to_array(img1)
            img1 = preprocess_input(img1)

            for interval in frame_intervals:
                if i + interval >= num_poses:
                    continue

                row2 = self.pose_df.iloc[i + interval]
                pos2 = np.array([row2['tx'], row2['ty'], row2['tz']])
                quat2 = np.array([row2['qx'], row2['qy'], row2['qz'], row2['qw']])
                # file_path2 = str(i + interval).zfill(6)
                # img_path2 = os.path.join(self.image_dir, f"{file_path2}.png")
                timestamp2 = str(int(float(row2['timestamp'])))
                img_path2 = os.path.join(self.image_dir, f"{timestamp2}.png")

                if not os.path.exists(img_path2):
                    print(f"File not found: {img_path2}")
                    continue

                img2 = load_img(img_path2, target_size=self.dim)
                img2 = img_to_array(img2)
                img2 = preprocess_input(img2)

                delta_pos = np.linalg.norm(pos2 - pos1)
                r1 = R.from_quat(quat1)
                r2 = R.from_quat(quat2)
                delta_rot = r2 * r1.inv()
                delta_angle = delta_rot.magnitude()

                if delta_pos > 0.05 or delta_angle > 6.2:
                    label = 1
                else:
                    label = 0

                # label = 1 if delta_pos > 0.1026 or delta_angle > 0.025 else 0
                deltas.append([f"{timestamp1}.png", f"{timestamp2}.png", delta_angle, delta_pos, label])
                
                X1.append(img1)
                X2.append(img2)
                y.append(label)

        deltas_df = pd.DataFrame(deltas, columns=['im1', 'im2', 'delta_angle', 'delta_pos', 'label'])
        deltas_df.to_csv(os.path.join(self.output_dir, "pose_deltas_euroc.csv"), index=False)
        print("Pose deltas CSV saved at:", os.path.join(self.output_dir, "pose_deltas_euroc.csv"))

        return np.array(X1), np.array(X2), np.array(y)


image_dir = '../Datasets/EuRoC/MH01/mav0/cam0/data'
pose_file = './evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt'

pose_processor = EuRoCDataset(image_dir, pose_file)
X1, X2, y = pose_processor.compute_deltas()
