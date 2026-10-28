import cv2
import os
import random

def generate_yolo_dataset(video_path, output_dir, start_sec, end_sec):
    print(f"🚀 Generating auto-annotated dataset from {start_sec}s to {end_sec}s...")
    
    dirs = {
        'train_img': os.path.join(output_dir, 'images/train'),
        'val_img': os.path.join(output_dir, 'images/val'),
        'train_lbl': os.path.join(output_dir, 'labels/train'),
        'val_lbl': os.path.join(output_dir, 'labels/val')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 因为有偏振片，干扰极小，阈值保持在 120
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=False)
    min_area = 1500
    
    frame_idx = start_frame
    saved_count = 0
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=3)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = (x + w/2) / width
                cy = (y + h/2) / height
                nw = w / width
                nh = h / height
                bboxes.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        if bboxes:
            is_val = random.random() < 0.2
            img_dir = dirs['val_img'] if is_val else dirs['train_img']
            lbl_dir = dirs['val_lbl'] if is_val else dirs['train_lbl']
            
            base_name = f"frame_{frame_idx:05d}"
            cv2.imwrite(os.path.join(img_dir, f"{base_name}.jpg"), frame)
            
            with open(os.path.join(lbl_dir, f"{base_name}.txt"), 'w') as f:
                f.write("\n".join(bboxes))
                
            saved_count += 1
            
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed up to frame {frame_idx}...")

    cap.release()
    
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: top_down_fish
"""
    yaml_path = os.path.join(output_dir, 'fish_dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\n✅ Dataset Generation Complete!")
    print(f"Total annotated images saved: {saved_count}")
    print(f"Dataset config created at: {yaml_path}")
    print("Ready for YOLOv8 training.")

if __name__ == "__main__":
    # 使用带偏振片的新素材，创建干净的新数据集
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1367.MOV'
    output = '/Users/qsw/Documents/fishcount/yolo_dataset_cpl'
    generate_yolo_dataset(video, output, 0, 32)
