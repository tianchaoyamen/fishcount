import cv2
import numpy as np
import pandas as pd
import os

def plot_trajectories(csv_path, output_path, width, height):
    df = pd.read_csv(csv_path)
    # 创建黑色背景
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 获取不同的 ID
    fish_ids = df['fish_id'].unique()
    
    # 为每个 ID 画出轨迹
    for fid in fish_ids:
        path = df[df['fish_id'] == fid]
        if len(path) < 15: continue # 过滤噪声
        
        points = path[['x', 'y']].values.astype(np.int32)
        # 用渐变色表示时间顺序（蓝色->红色）
        for i in range(1, len(points)):
            color = (int(255 * (i/len(points))), 0, int(255 * (1 - i/len(points))))
            cv2.line(canvas, tuple(points[i-1]), tuple(points[i]), color, 2)
            
    cv2.imwrite(output_path, canvas)
    print(f"Trajectory map saved to {output_path}")

if __name__ == "__main__":
    csv = '/Users/qsw/.gemini/antigravity/brain/fb54afa0-d53d-4846-a9bb-fcc396637f2e/scratch/fish_test_data.csv'
    out = '/Users/qsw/.gemini/antigravity/brain/fb54afa0-d53d-4846-a9bb-fcc396637f2e/scratch/trajectory_map.jpg'
    # 使用之前获取的分辨率 1080x1920
    plot_trajectories(csv, out, 1080, 1920)
