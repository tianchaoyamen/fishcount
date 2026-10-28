import cv2
import os
import numpy as np
from ultralytics import YOLO

class AntiDuplicateTroughCounter:
    def __init__(self, gate_pass=840, gate_ready=960, min_dist=75, max_lost=16, fps=60):
        self.gate_pass = gate_pass
        self.gate_ready = gate_ready
        self.min_dist = min_dist     # 物理排他性体宽间距 (小于此距离严禁产生新ID)
        self.max_lost = max_lost     # 水花遮挡记忆维持帧数 (0.27秒内不换ID)
        self.fps = fps
        
        self.next_id = 1
        self.tracks = {}
        self.counted_ids = set()
        self.event_log = []
        
    def update(self, detections, frame_idx):
        # detections: list of (x1, y1, x2, y2, conf)
        dets_sorted = sorted(detections, key=lambda d: d[4], reverse=True)
        
        # 1. 单帧防粘连多计: 高置信度优先，过滤体宽半径内的重复框
        clean_dets, clean_pts = [], []
        for d in dets_sorted:
            cx, cy = (d[0]+d[2])/2, (d[1]+d[3])/2
            if not any(np.hypot(cx - cp[0], cy - cp[1]) < self.min_dist for cp in clean_pts):
                clean_pts.append((cx, cy))
                clean_dets.append(d)
                
        # 2. 同鱼同轴长轴尾鳍抑制 (Head-to-Tail Suppression)
        # 若低置信度检出点与高置信度鱼头处在同一流道 (|dy| < 40px) 且间距在鱼身长度 (260~750px) 范围内，判定为鱼尾予以抑制
        is_tail = [False] * len(clean_dets)
        for i, (p1, d1) in enumerate(zip(clean_pts, clean_dets)):
            for j, (p2, d2) in enumerate(zip(clean_pts, clean_dets)):
                if i == j: continue
                if d2[4] >= 0.40 and d1[4] <= 0.35:
                    if abs(p1[1] - p2[1]) < 40 and 260 < abs(p1[0] - p2[0]) < 750:
                        is_tail[i] = True
                        break
                        
        valid_pts = [p for i, p in enumerate(clean_pts) if not is_tail[i]]
        valid_dets = [d for i, d in enumerate(clean_dets) if not is_tail[i]]
        
        assigned_tracks = set()
        assigned_dets = set()
        
        # 3. 匹配已有活跃轨迹 (流向椭圆加权与动态自适应流速外推)
        for tid, tinfo in list(self.tracks.items()):
            # 物理约束的自适应流速外推预测位置 (负值代表向左冲流)
            vx = max(-150.0, min(-25.0, tinfo['vx']))
            dt = tinfo['lost'] + 1
            pred_x = tinfo['pos'][0] + vx * dt
            pred_y = tinfo['pos'][1]
            
            best_idx = -1
            min_cost = 9999
            for i, (dx, dy) in enumerate(valid_pts):
                if i in assigned_dets: continue
                
                # 垂直方向严格限制 (急流中上下晃动不超 50px)
                if abs(dy - pred_y) > 50: continue
                # 顺流方向限制 (鱼不能逆水倒退超过 30px)
                if dx > tinfo['pos'][0] + 30: continue
                # 单帧物理极限位移约束 (单帧瞬时速度不能超 160px)
                if abs(dx - tinfo['pos'][0]) > 160 * dt: continue
                
                dist_x = abs(dx - pred_x)
                dist_y = abs(dy - pred_y)
                # 流向椭圆代价函数 (纵向严、横向宽容)
                cost = np.hypot(dist_x * 0.6, dist_y * 1.6)
                
                if dist_x < 180 and cost < min_cost:
                    min_cost = cost
                    best_idx = i
                    
            if best_idx != -1:
                assigned_tracks.add(tid)
                assigned_dets.add(best_idx)
                nx, ny = valid_pts[best_idx]
                ndet = valid_dets[best_idx]
                
                # 自适应流速更新 (指数滑动平均)
                dx_obs = (nx - tinfo['pos'][0]) / dt
                tinfo['vx'] = 0.6 * tinfo['vx'] + 0.4 * dx_obs
                tinfo['hits'] += 1
                tinfo['lost'] = 0
                tinfo['max_conf'] = max(tinfo['max_conf'], ndet[4])
                tinfo['history'].append((int(nx), int(ny)))
                tinfo['bbox'] = ndet
                
                # 4. 双区过线判定逻辑 (严格单向，单次终生闭锁)
                if tid not in self.counted_ids:
                    # 判定条件 A: 标准主门禁过线 (起点在门禁以右，冲过门禁线，累计位移 >= 60px，hits >= 3)
                    cond_std = (tinfo['start_pos'][0] >= self.gate_pass and nx < self.gate_pass and (tinfo['start_pos'][0] - nx >= 60) and tinfo['hits'] >= 3)
                    
                    # 判定条件 B: 极速大冲流冲线 (起点在门禁以右，冲过门禁线，单次位移 >= 100px，hits >= 2 且置信度高)
                    cond_fast = (tinfo['start_pos'][0] >= self.gate_pass and nx < self.gate_pass and (tinfo['start_pos'][0] - nx >= 100) and tinfo['hits'] >= 2 and tinfo['max_conf'] >= 0.45)
                    
                    # 判定条件 C: 中游出水补检门禁 (起点在门禁线下游 650~840 浮出，横贯滑动至出口 < 500，总位移 >= 180px，hits >= 4，max_conf >= 0.35)
                    cond_mid = (650 <= tinfo['start_pos'][0] < self.gate_pass and nx < 500 and (tinfo['start_pos'][0] - nx >= 180) and tinfo['hits'] >= 4 and tinfo['max_conf'] >= 0.35)
                    
                    if cond_std or cond_fast or cond_mid:
                        self.counted_ids.add(tid)
                        reason = "standard" if cond_std else ("fast_rush" if cond_fast else "mid_rescue")
                        self.event_log.append({
                            'frame': frame_idx,
                            'id': tid,
                            'pos': (int(nx), int(ny)),
                            'total': len(self.counted_ids),
                            'reason': reason
                        })
                        print(f"🐟 [{frame_idx/self.fps:5.2f}s | F{frame_idx:04d}] 鱼 ID #{tid:<3} 确认过线！累计: {len(self.counted_ids):<3} ({reason}, hits={tinfo['hits']}, vx={tinfo['vx']:.1f})")
                        
                tinfo['pos'] = (nx, ny)
            else:
                tinfo['lost'] += 1
                # 维持记忆 16 帧 (0.27秒)，水花重新露头后接续旧 ID
                if tinfo['lost'] > self.max_lost:
                    del self.tracks[tid]
                    
        # 5. 未分配的新检测点: 排他性空间与同体鱼身防重
        for i, (dx, dy) in enumerate(valid_pts):
            if i not in assigned_dets:
                # 距离已有活跃轨迹过近者禁止建号
                too_close = any(np.hypot(dx - t['pos'][0], dy - t['pos'][1]) < self.min_dist for t in self.tracks.values() if t['lost'] < 5)
                # 位于已有活跃鱼身长轴范围内 (|dy| < 35, |dx| < 280) 禁止建号
                too_close_body = any(abs(dy - t['pos'][1]) < 35 and abs(dx - t['pos'][0]) < 280 for t in self.tracks.values() if t['lost'] < 4)
                
                if not too_close and not too_close_body:
                    self.tracks[self.next_id] = {
                        'pos': (dx, dy),
                        'start_pos': (dx, dy),
                        'vx': -70.0, # 初始默认水槽流速
                        'hits': 1,
                        'lost': 0,
                        'max_conf': valid_dets[i][4],
                        'history': [(int(dx), int(dy))],
                        'bbox': valid_dets[i]
                    }
                    self.next_id += 1

def run_antiduplicate_pipeline(video_path, output_path, weights_path=None):
    print(f"🎬 启动物理防重/抗粘连高精度水槽鱼流分析引擎...")
    cap = cv2.VideoCapture(video_path)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    need_rotate = (raw_h > raw_w)
    out_w, out_h = (raw_h, raw_w) if need_rotate else (raw_w, raw_h)
    
    roi_y_min = int(out_h * 0.08)
    roi_y_max = int(out_h * 0.92)
    gate_pass = 840
    gate_ready = 960
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    
    if not weights_path or not os.path.exists(weights_path):
        if os.path.exists("weights/trough_head_best.pt"):
            weights_path = "weights/trough_head_best.pt"
        elif os.path.exists("runs/detect/trough_head/weights/best.pt"):
            weights_path = "runs/detect/trough_head/weights/best.pt"
        else:
            weights_path = "yolov8n.pt"
            
    model = YOLO(weights_path)
    tracker = AntiDuplicateTroughCounter(gate_pass=gate_pass, gate_ready=gate_ready, min_dist=75, max_lost=16, fps=fps)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if need_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        roi = frame[roi_y_min:roi_y_max, :]
        results = model(roi, conf=0.15, verbose=False, device='cpu')
        
        dets = []
        for b in results[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
            score = float(b.conf[0].item())
            dets.append((bx1, by1 + roi_y_min, bx2, by2 + roi_y_min, score))
            
        tracker.update(dets, frame_idx)
        
        vis = frame.copy()
        
        # 门禁线与准备线
        cv2.line(vis, (gate_ready, roi_y_min), (gate_ready, roi_y_max), (0, 165, 255), 2)
        cv2.putText(vis, "READY LINE", (gate_ready - 70, roi_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.line(vis, (gate_pass, roi_y_min), (gate_pass, roi_y_max), (0, 255, 0), 3)
        cv2.putText(vis, "PASS GATE", (gate_pass - 70, roi_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 活跃目标轨迹
        for tid, tinfo in tracker.tracks.items():
            if tinfo['lost'] > 0: continue
            bx1, by1, bx2, by2, score = tinfo['bbox']
            is_counted = (tid in tracker.counted_ids)
            color = (0, 255, 0) if is_counted else (0, 120, 255)
            label = f"#{tid} CONFIRMED" if is_counted else f"#{tid}"
            cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
            cv2.putText(vis, label, (int(bx1), max(20, int(by1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if len(tinfo['history']) > 1:
                pts = np.array(tinfo['history'][-20:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
            cv2.circle(vis, (int(tinfo['pos'][0]), int(tinfo['pos'][1])), 4, (0, 255, 255), -1)
            
        # HUD 仪表盘
        cv2.rectangle(vis, (30, 30), (580, 170), (20, 20, 20), -1)
        cv2.rectangle(vis, (30, 30), (580, 170), (180, 180, 180), 2)
        cv2.putText(vis, f"FISH COUNT: {len(tracker.counted_ids)}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(vis, f"Active Tracks: {len([t for t in tracker.tracks.values() if t['lost']==0])} | Anti-Dup Active",
                    (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(vis, f"Frame: {frame_idx}/{total_frames} ({frame_idx/fps:.1f}s)", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
                    
        writer.write(vis)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  处理进度: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%) | 纯净计数: {len(tracker.counted_ids)}...")
            
    cap.release()
    writer.release()
    
    report_file = os.path.splitext(output_path)[0] + "_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== 物理防重与抗粘连水槽鱼流分析报告 ===\n")
        f.write(f"视频素材: {video_path}\n")
        f.write(f"总帧数: {total_frames} ({total_frames/fps:.2f} 秒 @ {fps} fps)\n")
        f.write(f"去重后最终纯净鱼总数: {len(tracker.counted_ids)} 条\n")
        f.write("------------------------------------\n")
        f.write("全部确认通过事件流水 (前 50 条流水):\n")
        for ev in tracker.event_log[:50]:
            f.write(f"  第 {ev['frame']:04d} 帧 ({ev['frame']/fps:5.2f}s) | 鱼 ID #{ev['id']:<4} | 坐标: {ev['pos']} | 累计: {ev['total']}\n")
            
    print(f"\n==========================================")
    print(f"🎉 物理防重分析完成！")
    print(f"🐟 纯净统计总数: {len(tracker.counted_ids)} 条 (彻底消除隔帧重记与两鱼粘连多记)")
    print(f"📹 可视化视频: {output_path}")
    print(f"📄 纯净报表: {report_file}")
    print(f"==========================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="几何排他性单向终生闭锁水槽防重计数器")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--output", type=str, default="output/antidup_tracking.mp4", help="输出视频路径")
    parser.add_argument("--weights", type=str, default=None, help="YOLO 检测模型路径")
    args = parser.parse_args()

    video = args.video
    if not video or not os.path.exists(video):
        default_local = "新素材/IMG_2624.MOV"
        if os.path.exists(default_local):
            video = default_local
        else:
            print("⚠️ 请通过 --video 指定输入视频路径，例如: python antidup_tracker.py --video sample.mp4")
            exit(0)

    yolo_weights = args.weights
    if not yolo_weights or not os.path.exists(yolo_weights):
        if os.path.exists("weights/trough_head_best.pt"):
            yolo_weights = "weights/trough_head_best.pt"
        elif os.path.exists("runs/detect/trough_head/weights/best.pt"):
            yolo_weights = "runs/detect/trough_head/weights/best.pt"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    run_antiduplicate_pipeline(video, args.output, weights_path=yolo_weights)
