import cv2
import os

def extract_frames(video_path, output_dir, start_sec, end_sec, num_frames=30):
    print(f"🚀 Extracting {num_frames} frames from {start_sec}s to {end_sec}s...")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames_in_window = end_frame - start_frame
    
    # 计算抽帧间隔
    step = max(1, total_frames_in_window // num_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    saved_count = 0
    frame_idx = start_frame
    
    while frame_idx < end_frame and saved_count < num_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # 只有在正好达到间隔步长时才保存
        if (frame_idx - start_frame) % step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            
        frame_idx += 1

    cap.release()
    print(f"\n✅ Extraction Complete!")
    print(f"Saved {saved_count} high-quality frames to:\n👉 {output_dir}")

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1367.MOV'
    # 创建专门存放手工标注的文件夹
    output = '/Users/qsw/Documents/fishcount/manual_dataset/images/train'
    extract_frames(video, output, 22, 27, 30)
