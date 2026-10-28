import cv2
import os
import pandas as pd
from ultralytics import YOLO

def run_custom_yolo(video_path, output_dir, model_path, start_sec=None, end_sec=None):
    print(f"🚀 Loading your Manual custom trained YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    out_video = os.path.join(output_dir, f'tracking_{video_name}')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    start_frame = 0
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_sec is not None and end_sec is not None:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"🎯 Starting quick tracking test from {start_sec}s to {end_sec}s...")
    else:
        print("🎯 Starting full video tracking...")
    
    frame_idx = start_frame
    results_data = []
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        # 使用 CPU 避开 MPS 的空张量 Bug。置信度上调到 0.15 来消灭误识别。
        results = model.track(frame, persist=True, verbose=False, conf=0.15, device='cpu')
        
        annotated_frame = results[0].plot()
        
        unique_ids = set()
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for i, fid in enumerate(ids):
                cx = (boxes[i][0] + boxes[i][2]) / 2
                cy = (boxes[i][1] + boxes[i][3]) / 2
                results_data.append({"frame": frame_idx, "fish_id": fid, "x": cx, "y": cy})
            
            cv2.putText(annotated_frame, f"Current Unique IDs: {len(unique_ids)}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            
        writer.write(annotated_frame)
        frame_idx += 1
        
        if (frame_idx - start_frame) % 300 == 0:
            print(f"  Processed {frame_idx - start_frame} frames...")

    cap.release()
    writer.release()
    print(f"\n✅ Tracking Complete!")
    
    if results_data:
        df = pd.DataFrame(results_data)
        counts = {"CCW": 0, "CW": 0}
        
        for fid in df['fish_id'].unique():
            data = df[df['fish_id'] == fid]
            if len(data) < 15: continue 
            
            raw_pts = data[['x', 'y']].values
            pts = raw_pts[::5] 
            if len(pts) < 3: continue
            
            total_turn = 0
            for i in range(len(pts)-2):
                v1, v2 = pts[i+1]-pts[i], pts[i+2]-pts[i+1]
                total_turn += (v1[0]*v2[1] - v1[1]*v2[0])
            
            if total_turn > 0:
                counts["CCW"] += 1
            else:
                counts["CW"] += 1
                
        print(f"🐟 Total Valid Fish Tracks: {counts['CCW'] + counts['CW']}")
        print(f"🔄 Counter-Clockwise (CCW): {counts['CCW']}")
        print(f"🔃 Clockwise (CW): {counts['CW']}")
        
        report_txt = os.path.join(output_dir, f"report_{video_name}.txt")
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(f"=== 自动数鱼系统 分析报告 ===\n")
            f.write(f"分析视频文件: {video_path}\n")
            f.write(f"---------------------------------\n")
            f.write(f"🐟 有效鱼流总数 (Total): {counts['CCW'] + counts['CW']}\n")
            f.write(f"🔄 逆时针游动 (CCW) : {counts['CCW']}\n")
            f.write(f"🔃 顺时针游动 (CW)  : {counts['CW']}\n")
        print(f"📝 文字版分析报告已保存至: {report_txt}")
    
    print(f"📁 渲染视频已保存至: {out_video}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="鱼池环境 YOLOv8 跟踪与转向判定计数器")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--output", type=str, default="output/pond_tracking", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="YOLO 权重路径")
    parser.add_argument("--start_sec", type=float, default=None, help="起始时间(秒)")
    parser.add_argument("--end_sec", type=float, default=None, help="结束时间(秒)")
    args = parser.parse_args()

    # 模型权重自适应
    best_model = args.model
    if not best_model or not os.path.exists(best_model):
        if os.path.exists("weights/pond_manual_best.pt"):
            best_model = "weights/pond_manual_best.pt"
        elif os.path.exists("runs/detect/runs/detect/train_manual/weights/best.pt"):
            best_model = "runs/detect/runs/detect/train_manual/weights/best.pt"
        else:
            best_model = "yolov8n.pt"

    # 视频输入自适应
    video = args.video
    if not video or not os.path.exists(video):
        for candidate in ["素材/IMG_1368.MOV", "新素材/IMG_1368.MOV", "新素材/IMG_2624.MOV"]:
            if os.path.exists(candidate):
                video = candidate
                break
        if not video:
            print("⚠️ 请通过 --video 指定视频文件，例如: python custom_yolo_tracker.py --video sample.mp4")
            exit(0)

    output = args.output
    run_custom_yolo(video, output, best_model, start_sec=args.start_sec, end_sec=args.end_sec)
