import os
import torch
from ultralytics import YOLO

def train_fish_head_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(base_dir, "dataset_trough_head/data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"数据集配置文件未找到: {data_yaml}")
        
    # 检测加速设备 (MPS on Apple Silicon 或 CPU)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("⚡️ 检测到 Apple Silicon GPU，使用 MPS 设备加速训练！")
    else:
        device = 'cpu'
        print("💻 使用 CPU 设备进行训练。")
        
    print("🚀 加载预训练 YOLOv8n 权重...")
    model = YOLO("yolov8n.pt")
    
    output_project = "runs/detect"
    run_name = "trough_head"
    
    print(f"🏋️ 开始训练 (40 Epochs, imgsz=640)...")
    results = model.train(
        data=data_yaml,
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        project=output_project,
        name=run_name,
        exist_ok=True,
        fliplr=0.0,      # 关闭左右镜像翻转：让模型专注于水槽中鱼头朝右的固有特征
        flipud=0.0,      # 关闭上下翻转
        hsv_h=0.015,     # 轻微色调变化增强
        hsv_s=0.4,       # 饱和度增强
        hsv_v=0.4,       # 亮度增强（适应光照波动）
        workers=2,
        verbose=True
    )
    
    best_model_path = os.path.join(output_project, run_name, "weights", "best.pt")
    print(f"\n✅ 训练完成！最佳模型权重已保存至: {best_model_path}")
    return best_model_path

if __name__ == "__main__":
    train_fish_head_model()
