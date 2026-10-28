import cv2
import os
import numpy as np
from ultralytics import YOLO

class CalibratedTroughTracker:
    def __init__(self, gate_x=700, min_hits=3, max_dist=120):
        self.next_id = 1
        self.tracks = {} # id -> dict
        self.gate_x = gate_x
        self.min_hits = min_hits
        self.max_dist = max_dist
        self.counted_ids = set()
        self.event_log = []
        
    def update(self, detections, frame_idx):
        # detections: list of (x1, y1, x2, y2, conf)
        det_pts = [(((d[0]+d[2])/2), ((d[1]+d[3])/2)) for d in detections]
        
        assigned_tracks = set()
        assigned_dets = set()
        
        # 1. 轨迹匹配 (欧氏距离 + 水平向左位移先验)
        for tid, tinfo in list(self.tracks.items()):
            tx, ty = tinfo['pos']
            best_idx = -1
            min_score = self.max_dist
            
            for i, (dx, dy) in enumerate(det_pts):
                if i in assigned_dets: continue
                dist = np.hypot(dx - tx, dy - ty)
                # 鱼体向左滑行，如果反而大幅向右则施加位移惩罚
                if dx > tx + 20: dist += 35
                
                if dist < min_score:
                    min_score = dist
                    best_idx = i
                    
            if best_idx != -1:
                assigned_tracks.add(tid)
                assigned_dets.add(best_idx)
                nx, ny = det_pts[best_idx]
                
                tinfo['hits'] += 1
                tinfo['lost'] = 0
                tinfo['history'].append((int(nx), int(ny)))
                tinfo['bbox'] = detections[best_idx]
                
                # 越线状态机判定
                if tid not in self.counted_ids and tinfo['hits'] >= self.min_hits:
                    # 情况 A：从右侧向左穿越 gate_x
                    if tx >= self.gate_x and nx < self.gate_x:
                        self.counted_ids.add(tid)
                        self.event_log.append({
                            'frame': frame_idx,
                            'id': tid,
                            'y': int(ny),
                            'total': len(self.counted_ids),
                            'type': 'line_cross'
                        })
                        print(f"🐟 [Frame {frame_idx:04d}] 鱼 ID #{tid} 越线成功！当前总计: {len(self.counted_ids)}")
                    # 情况 B：初始在左侧或因急流在左侧首次检出的有效鱼，确认连续稳定追踪后计入
                    elif tinfo.get('initial_left', False):
                        self.counted_ids.add(tid)
                        tinfo['initial_left'] = False
                        self.event_log.append({
                            'frame': frame_idx,
                            'id': tid,
                            'y': int(ny),
                            'total': len(self.counted_ids),
                            'type': 'confirmed_flow'
                        })
                        print(f"🐟 [Frame {frame_idx:04d}] 鱼 ID #{tid} (左侧确认) 计入！当前总计: {len(self.counted_ids)}")
                        
                tinfo['pos'] = (nx, ny)
            else:
                tinfo['lost'] += 1
                if tinfo['lost'] > 5:
                    del self.tracks[tid]
                    
        # 2. 未分配目标分配新轨迹
        for i, (dx, dy) in enumerate(det_pts):
            if i not in assigned_dets:
                self.tracks[self.next_id] = {
                    'pos': (dx, dy),
                    'hits': 1,
                    'lost': 0,
                    'history': [(int(dx), int(dy))],
                    'bbox': detections[i],
                    'initial_left': (dx < self.gate_x) # 首次在门禁左侧出现，需 hits >= min_hits 确认
                }
                self.next_id += 1

def run_calibrated_counting(video_path, output_path, gate_x=700, conf=0.16):
    print(f"🎬 启动全视频校准水槽鱼流计数引擎 (v2)...")
    print(f"输入素材: {video_path}")
    print(f"门禁基准: X = {gate_x}, 检出阈值: {conf}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = YOLO("yolov8s-world.pt")
    model.set_classes(["fish"])
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    tracker = CalibratedTroughTracker(gate_x=gate_x, min_hits=3, max_dist=120)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 水槽有效区域 (y: 260~1180)
        roi = frame[260:1180, :]
        results = model(roi, conf=conf, verbose=False, device='cpu')
        
        dets = []
        for b in results[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
            score = float(b.conf[0].item())
            dets.append((bx1, by1 + 260, bx2, by2 + 260, score))
            
        tracker.update(dets, frame_idx)
        
        vis = frame.copy()
        
        # 1. 绘制虚拟计数门禁线 (高亮橙黄色)
        cv2.line(vis, (gate_x, 260), (gate_x, 1180), (0, 165, 255), 4)
        cv2.putText(vis, "COUNT GATE", (gate_x - 70, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # 2. 绘制活跃轨迹与目标识别框
        active_count = 0
        for tid, tinfo in tracker.tracks.items():
            if tinfo['lost'] > 0: continue
            active_count += 1
            bx1, by1, bx2, by2, score = tinfo['bbox']
            is_counted = (tid in tracker.counted_ids)
            
            color = (0, 255, 0) if is_counted else (0, 120, 255)
            label = f"#{tid} PASS" if is_counted else f"#{tid}"
            
            cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
            cv2.putText(vis, label, (int(bx1), max(20, int(by1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
            if len(tinfo['history']) > 1:
                pts = np.array(tinfo['history'][-20:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
            cv2.circle(vis, (int(tinfo['pos'][0]), int(tinfo['pos'][1])), 4, (0, 255, 255), -1)
            
        # 3. 顶部 HUD 仪表板
        cv2.rectangle(vis, (30, 30), (520, 150), (20, 20, 20), -1)
        cv2.rectangle(vis, (30, 30), (520, 150), (180, 180, 180), 2)
        cv2.putText(vis, f"FISH COUNT: {len(tracker.counted_ids)}", (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(vis, f"Active in View: {active_count} | Frame: {frame_idx}/{total_frames}", (50, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
                    
        writer.write(vis)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  处理进度: {frame_idx}/{total_frames} 帧 (当前累计: {len(tracker.counted_ids)} 条)...")
            
    cap.release()
    writer.release()
    
    # 提取 55 帧与 63 帧记录
    c55 = max([ev['total'] for ev in tracker.event_log if ev['frame'] <= 55] or [0])
    c63 = max([ev['total'] for ev in tracker.event_log if ev['frame'] <= 63] or [0])
    
    report_file = os.path.splitext(output_path)[0] + "_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== 水槽鱼流校准计数系统 分析报告 (v2) ===\n")
        f.write(f"视频文件: {video_path}\n")
        f.write(f"基准线位置: X={gate_x}\n")
        f.write(f"最终统计条数: {len(tracker.counted_ids)} 条\n")
        f.write(f"核心校准点对齐结果:\n")
        f.write(f"  - 第 55 帧累计: {c55} 条 (人工真值基准: 18 条)\n")
        f.write(f"  - 第 63 帧累计: {c63} 条 (人工真值基准: 23 条)\n")
        f.write("------------------------------------------\n")
        f.write("全部过线与确认事件流水:\n")
        for ev in tracker.event_log:
            sec = ev['frame'] / fps
            f.write(f"  第 {ev['frame']:04d} 帧 ({sec:.2f}s) | 鱼 ID #{ev['id']:<4} | 类型: {ev['type']} | 累计总数: {ev['total']}\n")
            
    print(f"\n==========================================")
    print(f"🎉 全视频校准处理完成！")
    print(f"🐟 统计鱼总数: {len(tracker.counted_ids)} 条")
    print(f"🎯 第 55 帧计数: {c55} (人工: 18)")
    print(f"🎯 第 63 帧计数: {c63} (人工: 23)")
    print(f"📹 结果视频: {output_path}")
    print(f"📄 详细报告: {report_file}")
    print(f"==========================================")
    return len(tracker.counted_ids), output_path

if __name__ == "__main__":
    video = "/Users/qsw/Documents/fishcount/新素材/IMG_2623.MOV"
    output = "/Users/qsw/Documents/fishcount/分析结果/trough_counted_v2.mp4"
    run_calibrated_counting(video, output, gate_x=700, conf=0.16)
