from ultralytics import YOLOWorld
import cv2
import os

def run_yolo_world(video_path, output_path, start_sec=31, end_sec=40):
    print("🚀 Initializing YOLO-World Model...")
    # 自动下载并加载 yolov8s-world 模型
    model = YOLOWorld("yolov8s-world.pt")
    
    # 零样本学习核心：用文本提示定义我们要找的物体
    model.set_classes(["fish"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"🎯 Tracking 'fish' using YOLO-World from {start_sec}s to {end_sec}s...")
    
    frame_idx = start_frame
    unique_ids = set()
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        # 执行推理与 ByteTrack 追踪
        # conf=0.1 是因为零样本检测置信度通常偏低，但对鱼类特征依然有效
        results = model.track(frame, persist=True, verbose=False, conf=0.1) 
        
        # 获取 YOLO 自动渲染的标注画面 (包含边界框和 ID)
        annotated_frame = results[0].plot()
        
        # 统计唯一 ID 数量
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            
            # 在画面左上角实时显示总过鱼量
            cv2.putText(annotated_frame, f"YOLO Total Unique Fish: {len(unique_ids)}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            
        writer.write(annotated_frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx - start_frame} frames...")

    cap.release()
    writer.release()
    print(f"\n✅ YOLO-World Analysis Complete!")
    print(f"🔥 Total Unique Fish Detected: {len(unique_ids)}")
    print(f"📁 Video saved to: {output_path}")
    print(f"💡 Suggestion: Watch the video to see how YOLO ignores the water ripples.")

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1338.MOV'
    output = '/Users/qsw/Documents/fishcount/分析结果/yolo/yolo_world_demo.mp4'
    run_yolo_world(video, output, 31, 40)
