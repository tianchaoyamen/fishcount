import cv2
import numpy as np
import pandas as pd
import os

class FishTracker:
    def __init__(self, video_path, output_dir, frame_limit=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_limit = frame_limit
        os.makedirs(output_dir, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 严格过滤参数
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=True)
        self.min_area = 1500  
        self.tracked_fish = {} 
        self.next_id = 0
        self.max_disappeared = 20 
        self.disappeared = {} 
        self.track_thresh = 300 

    def process(self):
        results = []
        frame_activity = [] # 记录每一帧的活跃度（检测到的轮廓数）
        
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = self.frame_limit if self.frame_limit else total_frames
        
        print(f"🚀 Final Analysis Start (With Smart Window Filtering): {os.path.basename(self.video_path)}")
        
        while frame_count < limit:
            ret, frame = self.cap.read()
            if not ret: break
            frame_count += 1
            
            fgmask = self.fgbg.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, kernel, iterations=3)
            
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_centroids = []
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_area:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        current_centroids.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
            
            # 记录这一帧有多少条潜在的鱼
            frame_activity.append({"frame": frame_count, "activity": len(current_centroids)})
            
            self._update_tracks(current_centroids)
            
            for fid, path in self.tracked_fish.items():
                if path:
                    results.append({"frame": frame_count, "fish_id": fid, "x": path[-1][0], "y": path[-1][1]})
            
            if frame_count % 500 == 0:
                print(f"  Progress: {frame_count}/{limit} frames...")

        self.cap.release()
        
        if results and frame_activity:
            self.finalize_analysis(pd.DataFrame(results), pd.DataFrame(frame_activity))

    def _update_tracks(self, centroids):
        if not self.tracked_fish:
            for c in centroids:
                self.tracked_fish[self.next_id], self.disappeared[self.next_id] = [c], 0
                self.next_id += 1
        else:
            ids = list(self.tracked_fish.keys())
            for fid in ids:
                if not centroids:
                    self.disappeared[fid] += 1
                else:
                    last = self.tracked_fish[fid][-1]
                    dists = [np.linalg.norm(np.array(c)-np.array(last)) for c in centroids]
                    idx = np.argmin(dists)
                    if dists[idx] < self.track_thresh:
                        self.tracked_fish[fid].append(centroids[idx])
                        self.disappeared[fid] = 0
                        centroids.pop(idx)
                    else:
                        self.disappeared[fid] += 1
                
                if self.disappeared.get(fid, 0) > self.max_disappeared:
                    del self.tracked_fish[fid]
                    del self.disappeared[fid]

            for c in centroids:
                self.tracked_fish[self.next_id], self.disappeared[self.next_id] = [c], 0
                self.next_id += 1

    def finalize_analysis(self, df, activity_df):
        print("\n" + "="*50)
        print("🧠 SMART FILTERING & FINAL REPORT")
        print("="*50)
        
        # 1. 寻找鱼群活跃的高峰期 (滑动窗口平滑)
        window_size = int(self.fps * 2) # 2秒滑动窗口
        activity_df['smooth_activity'] = activity_df['activity'].rolling(window=window_size, center=True).mean()
        
        # 定义活跃阈值（画面中平均存在超过 1 条鱼被视为高活跃期）
        active_frames = activity_df[activity_df['smooth_activity'] > 1.0]['frame'].tolist()
        
        if not active_frames:
            print("❌ No clear fish swarm window detected. Outputting raw data.")
            active_frames = activity_df['frame'].tolist()
        else:
            start_sec = min(active_frames) / self.fps
            end_sec = max(active_frames) / self.fps
            print(f"🎯 Detected Fish Swarm Event: ~{start_sec:.1f}s to ~{end_sec:.1f}s")
        
        # 2. 过滤数据
        counts = {"CCW": 0, "CW": 0}
        valid_ids = []
        
        for fid in df['fish_id'].unique():
            data = df[df['fish_id'] == fid]
            
            # 条件A: 这条鱼的轨迹必须与我们检测到的“活跃期”有交集 (丢弃无鱼时段的噪音)
            track_frames = data['frame'].tolist()
            if not set(track_frames).intersection(active_frames):
                continue
            
            # 条件B: 存活时间大于 15 帧（0.5秒）
            if len(data) < 15: continue 
            
            raw_pts = data[['x', 'y']].values
            
            # 条件C: 明显的位移 (大于150像素)
            displacement = np.linalg.norm(raw_pts[-1] - raw_pts[0])
            if displacement < 150: continue

            # 通过所有考核，开始判定方向
            valid_ids.append(fid)
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
        
        print(f"Total Valid Fish Tracks (Filtered): {len(valid_ids)}")
        print(f"🔄 Counter-Clockwise (CCW): {counts['CCW']}")
        print(f"🔃 Clockwise (CW): {counts['CW']}")
        print("="*50)
        
        # 导出详细 CSV
        filtered_df = df[df['fish_id'].isin(valid_ids)]
        csv_path = os.path.join(self.output_dir, 'final_smart_fish_count.csv')
        filtered_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1338.MOV'
    output = '/Users/qsw/Documents/fishcount/分析结果/'
    FishTracker(video, output, frame_limit=None).process()
