##################################################################
# Analyze keyframe ground truth pose
##################################################################


import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns

def read_keyframe_file(file_path):
    keyframes = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 8:
                timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, data)
                keyframes.append((timestamp, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))
    return keyframes

def compute_delta_pose(kf1, kf2):
    pos1, quat1 = kf1[1], kf1[2]
    pos2, quat2 = kf2[1], kf2[2]
    
    delta_pos = pos2 - pos1
    delta_pos_magnitude = np.linalg.norm(delta_pos)
    
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    delta_rot = r2 * r1.inv()
    delta_angle = delta_rot.magnitude()
    
    return delta_pos_magnitude, delta_angle

def main(file_path):
    keyframes = read_keyframe_file(file_path)
    delta_poses = []

    for i in range(len(keyframes) - 1):
        kf1 = keyframes[i]
        kf2 = keyframes[i + 1]
        delta_pos_magnitude, delta_angle = compute_delta_pose(kf1, kf2)
        
        delta_poses.append([kf1[0], kf2[0], delta_pos_magnitude, delta_angle])

    df = pd.DataFrame(delta_poses, columns=['timestamp1', 'timestamp2', 'delta_pos', 'delta_angle'])
    return df

file_path = 'Examples/KeyFrameTrajectory.txt'
df = main(file_path)

# 트랜슬레이션과 회전 변화의 통계
translation_threshold = 0.1
rotation_threshold = 0.1  # 라디안 단위 (약 5.7도)

translation_stats = df['delta_pos'].describe()
rotation_stats = df['delta_angle'].describe()

print("### 트랜슬레이션 변화 (Translation Change)")
print(f"- **평균**: {translation_stats['mean']:.4f} 미터")
print(f"- **표준편차**: {translation_stats['std']:.4f}")
print(f"- **최소값**: {translation_stats['min']:.6f} 미터")
print(f"- **25% 백분위수**: {translation_stats['25%']:.4f} 미터")
print(f"- **중앙값 (50% 백분위수)**: {translation_stats['50%']:.4f} 미터")
print(f"- **75% 백분위수**: {translation_stats['75%']:.4f} 미터")
print(f"- **최대값**: {translation_stats['max']:.4f} 미터")
print(f"\n대부분의 트랜슬레이션 변화는 {translation_stats['mean']:.4f} 미터(약 {translation_stats['mean'] * 100:.1f}cm) 정도의 변화를 가지며, 최대 변화량은 {translation_stats['max']:.4f} 미터입니다. {len(df[df['delta_pos'] > translation_threshold])}개의 변화가 {translation_threshold} 미터를 초과했습니다.\n")

print("### 회전 변화 (Rotation Change)")
print(f"- **평균**: {rotation_stats['mean']:.4f} 라디안 (약 {np.degrees(rotation_stats['mean']):.1f}도)")
print(f"- **표준편차**: {rotation_stats['std']:.4f} 라디안")
print(f"- **최소값**: {rotation_stats['min']:.4f} 라디안")
print(f"- **25% 백분위수**: {rotation_stats['25%']:.4f} 라디안")
print(f"- **중앙값 (50% 백분위수)**: {rotation_stats['50%']:.4f} 라디안")
print(f"- **75% 백분위수**: {rotation_stats['75%']:.4f} 라디안")
print(f"- **최대값**: {rotation_stats['max']:.4f} 라디안")
print(f"\n회전 변화는 대체로 큰 값을 가지며, 평균적으로 약 {rotation_stats['mean']:.4f} 라디안(약 {np.degrees(rotation_stats['mean']):.1f}도)의 변화를 보입니다. 이 결과는 키프레임 사이에서 큰 회전 변화가 자주 발생한다는 것을 나타냅니다. {len(df[df['delta_angle'] > rotation_threshold])}개의 변화가 {rotation_threshold} 라디안을 초과했습니다.\n")

# 트랜슬레이션과 회전 변화의 히스토그램
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['delta_pos'], kde=True)
plt.title('Delta Translation Magnitude')

plt.subplot(1, 2, 2)
sns.histplot(df['delta_angle'], kde=True)
plt.title('Delta Rotation Magnitude (radians)')

plt.tight_layout()
plt.show()
