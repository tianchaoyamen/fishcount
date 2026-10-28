import cv2
import os

video_path = '/Users/qsw/Documents/fishcount/素材/IMG_1338.MOV'
output_dir = '/Users/qsw/.gemini/antigravity/brain/fb54afa0-d53d-4846-a9bb-fcc396637f2e/scratch'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total Frames: {total_frames}")

# Jump to frame 50
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()

if ret:
    frame_save_path = os.path.join(output_dir, 'preview_frame.jpg')
    cv2.imwrite(frame_save_path, frame)
    print(f"Saved preview frame to {frame_save_path}")
else:
    print("Error: Could not read frame 50")

cap.release()
