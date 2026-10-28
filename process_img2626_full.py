import cv2
import os
import time
import torch
import numpy as np
from spatiotemporal_flow_engine import SpatioTemporalFlowEngine

def run_img2626_full_analysis(video_path, output_mp4, report_path):
    print(f"🎬 启动 IMG_2626.MOV 全视频时空连续流分析管道...")
    print(f"输入素材: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    need_rotate = (raw_h > raw_w)
    out_w, out_h = (raw_h, raw_w) if need_rotate else (raw_w, raw_h)
    
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    writer = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    
    engine = SpatioTemporalFlowEngine(fps=fps)
    
    print(f"视频时长: {total_frames/fps:.2f} 秒 ({total_frames} 帧 @ {fps}fps)")
    t0 = time.time()
    
    fidx = 0
    checkpoint_interval = 300 # 每 5 秒输出一次心跳
    
    milestone_secs = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, int(total_frames/fps)]
    milestone_frames = set([s * fps for s in milestone_secs] + [total_frames - 1])
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if need_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        cum_total, inst_view, w_surge, inst_flux, dmap, dets = engine.process_frame(frame)
        
        vis = frame.copy()
        
        # 1. 密度热力图半透明渲染
        dmap_norm = (dmap / (np.max(dmap) + 1e-6) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        heatmap_roi = cv2.resize(heatmap, (out_w, 900))
        vis_roi = vis[80:980, :]
        
        alpha_blend = 0.15 + 0.25 * w_surge
        vis[80:980, :] = cv2.addWeighted(vis_roi, 1.0 - alpha_blend, heatmap_roi, alpha_blend, 0)
        
        # 2. 门禁线绘制
        gate_x = 840
        gate_color = (0, 0, 255) if w_surge > 0.5 else (0, 255, 0)
        cv2.line(vis, (gate_x, 80), (gate_x, 980), gate_color, 4)
        gate_label = "SURGE FLUX GATE" if w_surge > 0.5 else "DISCRETE PASS GATE"
        cv2.putText(vis, gate_label, (gate_x - 110, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gate_color, 2)
        
        # 3. 全流态绘制 YOLO 目标检测框与轨迹 (消除盲区，全生命周期可视化)
        for tid, tinfo in engine.tracker.tracks.items():
            if tinfo['lost'] > 0: continue
            bx1, by1, bx2, by2, score = tinfo['bbox']
            is_counted = (tid in engine.tracker.counted_ids)
            bcolor = (0, 255, 0) if is_counted else (0, 165, 255)
            thick = 3 if is_counted else 2
            cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), bcolor, thick)
            
            label = f"#{tid} CONFIRMED" if is_counted else f"#{tid} ({score:.2f})"
            # 绘制文字底色标签增强对比度
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (int(bx1), max(20, int(by1) - th - 8)), (int(bx1) + tw + 6, max(20, int(by1))), (20, 20, 20), -1)
            cv2.putText(vis, label, (int(bx1) + 3, max(20, int(by1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcolor, 2)
            
            # 绘制轨迹线
            if len(tinfo['history']) > 1:
                pts = np.array(tinfo['history'][-20:], dtype=np.int32).reshape((-1, 1, 2))
                tcolor = (0, 255, 255) if is_counted else (0, 140, 255)
                cv2.polylines(vis, [pts], False, tcolor, 2)
                cv2.circle(vis, (int(tinfo['pos'][0]), int(tinfo['pos'][1])), 4, (0, 255, 255), -1)
                            
        # 4. HUD 顶部实时仪表盘
        cv2.rectangle(vis, (30, 25), (740, 205), (15, 15, 15), -1)
        cv2.rectangle(vis, (30, 25), (740, 205), (180, 180, 180), 2)
        
        cv2.putText(vis, f"TOTAL FISH: {int(round(cum_total))} ({cum_total:.1f})", (50, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                    
        regime_text = "[SURGE FLUX REGIME]" if w_surge > 0.5 else "[DISCRETE TRACKING REGIME]"
        regime_color = (0, 140, 255) if w_surge > 0.5 else (0, 255, 120)
        cv2.putText(vis, regime_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, regime_color, 2)
        
        cv2.putText(vis, f"Fish in View: {inst_view:.1f} | Flux: {inst_flux*fps:.1f} fish/s | Blend: {w_surge*100:.0f}%",
                    (50, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                    
        # 最新过线事件提示
        last_ev = engine.tracker.event_log[-1] if len(engine.tracker.event_log) > 0 else None
        if last_ev and (fidx - last_ev['frame'] < 45): # 闪烁显示 0.75 秒
            ev_text = f"PASS EVENT: Fish #{last_ev['id']} Confirmed! Total: {last_ev['total']}"
            cv2.putText(vis, ev_text, (50, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(vis, f"Time: {fidx/fps:5.2f}s ({fidx}/{total_frames}) | Video: IMG_2626 (Full)",
                        (50, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1)
                        
        cv2.putText(vis, f"Active Tracks: {len([t for t in engine.tracker.tracks.values() if t['lost']==0])} | Pure Counted: {len(engine.tracker.counted_ids)}",
                    (50, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
                    
        writer.write(vis)
        
        if fidx in milestone_frames or fidx % checkpoint_interval == 0:
            print(f"  进度 [{fidx/fps:5.1f}s | F{fidx:04d}/{total_frames} ({fidx/total_frames*100:5.1f}%)]: 累计通过={cum_total:5.1f} (整型: {int(round(cum_total)):3d}), 在视={inst_view:4.1f}, 软权重={w_surge*100:3.0f}%, 模态={regime_text[:14]}", flush=True)
            
        fidx += 1
        
    cap.release()
    writer.release()
    t1 = time.time()
    
    h = engine.history
    final_count = h[-1]['cum_total'] if len(h) > 0 else 0.0
    final_yolo = h[-1]['yolo_raw'] if len(h) > 0 else 0
    
    # 统计涌浪时间占比
    surge_frames = sum(1 for item in h if item['w_surge'] > 0.5)
    discrete_frames = len(h) - surge_frames
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== IMG_2626.MOV 全视频时空连续流通量分析报告 ===\n")
        f.write(f"视频素材: {video_path}\n")
        f.write(f"总时长: {total_frames/fps:.2f} 秒 ({total_frames} 帧 @ {fps} fps)\n")
        f.write(f"实际计算耗时: {t1-t0:.1f} 秒 (平均处理速度: {total_frames/(t1-t0):.1f} fps)\n")
        f.write(f"人工逐帧全视频真值: 282 条\n")
        f.write(f"最终累计总通过鱼数: {int(round(final_count))} 条 (浮点实数积分: {final_count:.2f} 条)\n")
        err = abs(final_count - 282)
        acc = max(0.0, 100.0 - err / 282.0 * 100.0)
        f.write(f"真值吻合度: {acc:.1f}% (绝对误差: {err:.1f} 条)\n")
        f.write(f"纯 2D YOLO 原始检测数: {final_yolo} 条\n")
        f.write(f"时空连续流通量补偿增量: +{final_count - final_yolo:.1f} 条 (+{(final_count - final_yolo)/final_yolo*100:.1f}%)\n")
        f.write(f"流态时间分布: ⚡ 密集涌浪期占 {surge_frames/total_frames*100:.1f}% ({surge_frames/fps:.1f}s) | 🟢 离散稀疏期占 {discrete_frames/total_frames*100:.1f}% ({discrete_frames/fps:.1f}s)\n")
        f.write("------------------------------------\n")
        f.write("时间轴关键节点累计过线流水:\n")
        for s in milestone_secs:
            mf = min(s * fps, total_frames - 1)
            if mf < len(h):
                item = h[mf]
                mode_str = "密集涌浪 (通量积分)" if item['w_surge'] > 0.5 else "稀疏单鱼 (防重追踪)"
                f.write(f"  - 第 {s:4.1f}s (F{mf:04d}): 累计 = {item['cum_total']:5.1f} 条 (整型: {int(round(item['cum_total'])):3d}) | 瞬时在视 = {item['inst_view']:4.1f} 条 | 模式 = {mode_str}\n")
        f.write("------------------------------------\n")
        f.write("全片物理流态与算法特征说明:\n")
        f.write("  1. 无需安装任何可能割伤鱼体的物理隔板或漏斗收口，实现 100% 活鱼无损安全。\n")
        f.write("  2. 在前期大通量涌浪期，鱼体立体叠罗汉并伴随白水浪花，系统自适应启用 3D 深度扩张截面通量积分，克服 NMS 抹杀重叠框的固有缺陷。\n")
        f.write("  3. 在中后期稀疏细流期，系统平滑回归单向终生闭锁几何追踪，彻底杜绝回冲多计与重复计数。\n")

    print("\n" + "="*50)
    print(f"🎉 IMG_2626.MOV 全视频分析渲染完成！")
    print(f"🐟 全片累计总鱼数: {int(round(final_count))} 条 (精确实数: {final_count:.1f} 条)")
    print(f"📊 纯 YOLO 对比: {final_yolo} 条 | 通量补偿增量: +{final_count - final_yolo:.1f} 条")
    print(f"📹 渲染输出视频: {output_mp4}")
    print(f"📄 详尽分析报告: {report_path}")
    print(f"⏱️ 耗时: {t1-t0:.1f}s (处理速度: {total_frames/(t1-t0):.1f} fps)")
    print("="*50)

if __name__ == "__main__":
    video = "/Users/qsw/Documents/fishcount/新素材/IMG_2626.MOV"
    output_mp4 = "/Users/qsw/Documents/fishcount/分析结果/IMG_2626_spatiotemporal_flow.mp4"
    report = "/Users/qsw/Documents/fishcount/分析结果/IMG_2626_report.txt"
    run_img2626_full_analysis(video, output_mp4, report)
