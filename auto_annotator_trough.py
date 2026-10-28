import cv2
import os
import shutil
import numpy as np
from ultralytics import YOLO

def generate_trough_head_dataset(video_path, output_dir, step=2):
    print(f"🚀 启动基于 YOLO-World 引导 + 背景轮廓端点的高精度鱼头自动标注流水线...")
    print(f"视频路径: {video_path}")
    print(f"输出目录: {output_dir}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    dirs = {
        'train_img': os.path.join(output_dir, 'images/train'),
        'val_img': os.path.join(output_dir, 'images/val'),
        'train_lbl': os.path.join(output_dir, 'labels/train'),
        'val_lbl': os.path.join(output_dir, 'labels/val'),
        'vis': os.path.join(output_dir, 'samples_vis')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # 加载预训练引导模型
    teacher = YOLO('yolov8s-world.pt')
    teacher.set_classes(['fish'])
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 提取干净背景帧 (140~155 帧无鱼)
    bg_frames = []
    for f in [140, 145, 150, 155]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, f_img = cap.read()
        if ret:
            bg_frames.append(f_img.astype(np.float32))
    bg_img = np.median(bg_frames, axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    
    # 水槽 ROI: 排除上下金属反光条
    roi_y1, roi_y2 = 260, 1180
    
    saved_images = 0
    total_heads = 0
    
    for f_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.subtract(bg_gray, frame_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # YOLO-World 在 ROI 内检测完整鱼体
        roi = frame[roi_y1:roi_y2, :]
        results = teacher(roi, conf=0.22, iou=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        heads = []
        vis = frame.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            y1 += roi_y1
            y2 += roi_y1
            
            # 过滤过小或边缘截断严重的框
            w = x2 - x1
            h = y2 - y1
            if w < 120 or h < 60:
                continue
            if x2 > width - 15: # 鱼刚进画面，头部可能还不全
                continue
                
            crop_thresh = thresh[y1:y2, x1:x2]
            cnts, _ = cv2.findContours(crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 3000: continue
            
            # 鱼头在右侧前端极点
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            abs_rx = x1 + rightmost[0]
            abs_ry = y1 + rightmost[1]
            
            if abs_rx > width - 30 or abs_rx < 60:
                continue
                
            # 鱼头标准尺寸
            hw = 125
            hh = 105
            hx2 = min(width - 5, abs_rx + 8)
            hx1 = max(5, hx2 - hw)
            hy1 = max(roi_y1, abs_ry - hh // 2)
            hy2 = min(roi_y2, hy1 + hh)
            
            # 质量过滤：避免框内全是水花或空背景
            head_crop = frame_gray[int(hy1):int(hy2), int(hx1):int(hx2)]
            if head_crop.size == 0: continue
            
            # 去重检查 (IOU > 0.35 则合并/忽略)
            duplicate = False
            for ex1, ey1, ex2, ey2 in heads:
                ix1, iy1 = max(hx1, ex1), max(hy1, ey1)
                ix2, iy2 = min(hx2, ex2), min(hy2, ey2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (hx2 - hx1)*(hy2 - hy1) + (ex2 - ex1)*(ey2 - ey1) - inter
                    if inter / union > 0.35:
                        duplicate = True
                        break
            if duplicate: continue
            
            heads.append((hx1, hy1, hx2, hy2))
            cv2.rectangle(vis, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 0, 255), 3)
            cv2.putText(vis, 'FishHead', (int(hx1), max(20, int(hy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        if heads:
            is_val = (saved_images % 5 == 0)
            img_dest = dirs['val_img'] if is_val else dirs['train_img']
            lbl_dest = dirs['val_lbl'] if is_val else dirs['train_lbl']
            
            fname = f"trough_{f_idx:04d}"
            cv2.imwrite(os.path.join(img_dest, f"{fname}.jpg"), frame)
            
            yolo_lines = []
            for hx1, hy1, hx2, hy2 in heads:
                cx = ((hx1 + hx2) / 2.0) / width
                cy = ((hy1 + hy2) / 2.0) / height
                nw = (hx2 - hx1) / float(width)
                nh = (hy2 - hy1) / float(height)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
            with open(os.path.join(lbl_dest, f"{fname}.txt"), 'w') as f:
                f.write("\n".join(yolo_lines))
                
            if saved_images < 25:
                cv2.imwrite(os.path.join(dirs['vis'], f"vis_{fname}.jpg"), vis)
                
            saved_images += 1
            total_heads += len(heads)
            
    cap.release()
    
    yaml_text = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: fish_head
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)
        
    print(f"\n🎉 自动标注完成！")
    print(f"📊 保存总图像数: {saved_images} (训练: {saved_images - saved_images//5}, 验证: {saved_images//5})")
    print(f"🐟 标注鱼头目标数: {total_heads}")
    print(f"📁 数据集路径: {output_dir}")
    print(f"📄 配置文件: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    video = "/Users/qsw/Documents/fishcount/新素材/IMG_2623.MOV"
    output = "/Users/qsw/Documents/fishcount/dataset_trough_head"
    generate_trough_head_dataset(video, output, step=2)
