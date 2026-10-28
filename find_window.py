import cv2
import os

def find_activity_window(video_path):
    print(f"🔍 Scanning {os.path.basename(video_path)} for fish activity...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 稍微放宽一点阈值，确保能捕捉到
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=80, detectShadows=False)
    min_area = 1000 
    
    frame_count = 0
    activity_log = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fish_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                fish_count += 1
                
        if fish_count > 0:
            activity_log.append(frame_count / fps)
            
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"  Scanned {frame_count} frames...")

    cap.release()
    
    if activity_log:
        start_time = min(activity_log)
        end_time = max(activity_log)
        print(f"\n✅ Activity Detected!")
        print(f"Fish are present roughly between {start_time:.1f}s and {end_time:.1f}s")
        return start_time, end_time
    else:
        print("\n❌ No significant activity found.")
        return None, None

if __name__ == "__main__":
    video = '/Users/qsw/Documents/fishcount/素材/IMG_1367.MOV'
    find_activity_window(video)
