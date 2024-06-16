import pandas as pd

# CSV 파일 경로 설정
csv_file_path = 'delta_pose_delta_rotation.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# delta_pose와 delta_rotation의 평균과 분산 계산
delta_pose_mean = df['delta_pose'].mean()
delta_pose_var = df['delta_pose'].var()

delta_rotation_mean = df['delta_rotation'].mean()
delta_rotation_var = df['delta_rotation'].var()

# 결과 출력
print(f"Delta Pose - Mean: {delta_pose_mean}, Variance: {delta_pose_var}")
print(f"Delta Rotation - Mean: {delta_rotation_mean}, Variance: {delta_rotation_var}")
