import cv2
import numpy as np
import pandas as pd
import os

class PreciseFishCounter:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 使用黄金参数
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=80, detectShadows=False)
        self.min_area = 1000 
        
    def run_window(self, start_sec, end_sec):
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(self.output_dir, 'precise_check_1367.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
        
        fish_data = []
        frame_idx = start_frame
        
        print(f"🔍 Analyzing specific window: {start_sec}s to {end_sec}s...")
        
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret: break
            
            fgmask = self.fgbg.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, kernel, iterations=3)
            
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            fish_this_frame = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.min_area:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0: continue
                    
                    # 引入圆度过滤倒影
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.15:
                        fish_this_frame += 1
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(frame, f"Fish C:{circularity:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            cv2.putText(frame, f"Time: {frame_idx/self.fps:.2f}s  Count: {fish_this_frame}", (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            
            writer.write(frame)
            fish_data.append({"frame": frame_idx, "count": fish_this_frame})
            frame_idx += 1

        self.cap.release()
        writer.release()
        
        df = pd.DataFrame(fish_data)
        max_overlap = df['count'].max()
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {out_path}")
        print(f"Max concurrent fish in one frame: {max_overlap}")

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1367.MOV'
    output = '/Users/qsw/Documents/fishcount/分析结果/精准分析/'
    # 分析你指定的 22秒 到 27秒 
    counter = PreciseFishCounter(video, output)
    counter.run_window(22, 27)
