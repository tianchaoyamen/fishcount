import os
import glob
from ultralytics import YOLO

def train_blackfish_model():
    print("🚀 准备启动黑鱼鱼头高精度微调训练...")
    
    # 检查是否有用户标注的 labels
    label_files = glob.glob("/Users/qsw/Documents/fishcount/待标注_急流黑鱼10张/labels/*.txt")
    print(f"找到标注文件: {len(label_files)} 个")
    
    if len(label_files) == 0:
        print("⚠️ 未检测到标注文件，请先在 '待标注_急流黑鱼10张/labels' 下放置 YOLO 格式标注！")
        return
        
    dataset_yaml = "/Users/qsw/Documents/fishcount/待标注_急流黑鱼10张/blackfish_data.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"""
path: /Users/qsw/Documents/fishcount/待标注_急流黑鱼10张
train: .
val: .
names:
  0: fish_head
""")
    print("已生成数据集配置文件:", dataset_yaml)
    
    # 加载预训练模型继续微调 30 轮
    model = YOLO("runs/detect/trough_head/weights/best.pt")
    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=4,
        project="runs/detect",
        name="blackfish_head_finetuned",
        exist_ok=True,
        verbose=True
    )
    print("🎉 黑鱼鱼头微调训练完成！新权重保存在: runs/detect/blackfish_head_finetuned/weights/best.pt")

if __name__ == "__main__":
    train_blackfish_model()
