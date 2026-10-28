import cv2
import numpy as np
import os

def extract_and_filter(video_path, output_path, start_sec, end_sec):
    print(f"🚀 Extracting and filtering video from {start_sec}s to {end_sec}s...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 背景建模
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=80, detectShadows=False)
    
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=3)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fish_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000: # 1. 面积门槛
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                
                # 2. 圆度过滤 (核心抗干扰逻辑)
                # 圆度计算公式: 4 * pi * Area / Perimeter^2
                # 线条的圆度接近于 0，圆形的圆度为 1，鱼通常在 0.2 - 0.7 之间
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # 如果是线条倒影，将被过滤掉
                if circularity > 0.15:
                    fish_count += 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # 绿色框代表通过了形状校验，被认为是真正的鱼
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.putText(frame, f"Fish C:{circularity:.2f}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    # 红色框代表这是倒影/线条干扰，已被算法忽略
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Reflection", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Time: {frame_idx/fps:.1f}s | Fish in frame: {fish_count}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n✅ Extraction and Filtering Complete!")
    print(f"📁 Please check the video: {output_path}")

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1367.MOV'
    output = '/Users/qsw/Documents/fishcount/分析结果/精准分析/clip_22s_27s_filtered.mp4'
    extract_and_filter(video, output, 22, 27)
