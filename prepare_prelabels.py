import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

# 为 10 张待标注图生成预标注
img_dir = "/Users/qsw/Documents/fishcount/待标注_急流黑鱼10张"
vis_dir = os.path.join(img_dir, "prelabels_vis")
label_dir = os.path.join(img_dir, "labels")
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

model = YOLO("runs/detect/trough_head/weights/best.pt")

img_files = sorted(glob.glob(os.path.join(img_dir, "frame_*.jpg")))
print(f"Found {len(img_files)} images for pre-labeling.")

for img_path in img_files:
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # 水槽区域
    roi = img[80:980, :]
    results = model(roi, conf=0.04, verbose=False, device='cpu')
    
    boxes = []
    for b in results[0].boxes:
        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
        score = float(b.conf[0].item())
        boxes.append((bx1, by1 + 80, bx2, by2 + 80, score))
        
    # 空间抑制: 距离过近合并
    clean_boxes = []
    for b in sorted(boxes, key=lambda x: x[4], reverse=True):
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        if not any(np.hypot(cx - (cb[0]+cb[2])/2, cy - (cb[1]+cb[3])/2) < 45 for cb in clean_boxes):
            clean_boxes.append(b)
            
    # 输出 YOLO txt
    basename = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, f"{basename}.txt")
    with open(txt_path, "w") as f:
        for b in clean_boxes:
            bx1, by1, bx2, by2, _ = b
            bw = (bx2 - bx1) / w
            bh = (by2 - by1) / h
            bcx = ((bx1 + bx2) / 2) / w
            bcy = ((by1 + by2) / 2) / h
            # 类别 0: fish_head
            f.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
            
    # 可视化输出
    vis = img.copy()
    for b in clean_boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(vis_dir, f"vis_{basename}.jpg"), vis)
    print(f"  {basename}: generated {len(clean_boxes)} pre-labeled head boxes.")

print("Pre-labeling complete! Check 'prelabels_vis' folder.")
