import cv2
import os
import torch
import numpy as np
from density_model import FishDensityNet

def run_density_video_pipeline(video_path, output_path):
    print("🎬 启动全量 211 点高精度点密度网络视频分析引擎...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = FishDensityNet().to(device)
    model.load_state_dict(torch.load("runs/density/fish_density_full211.pth", map_location=device))
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    cap = cv2.VideoCapture(video_path)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    need_rotate = (raw_h > raw_w)
    out_w, out_h = (raw_h, raw_w) if need_rotate else (raw_w, raw_h)
    
    roi_y_min = int(out_h * 0.08)
    roi_y_max = int(out_h * 0.92)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    
    # 门禁带参数
    gate_x_px = 880
    gate_band_half = 25 # 宽度 50 像素
    
    # 门禁通量自适应累积
    cum_passage_count = 0.0
    passage_history = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if need_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        roi = frame[roi_y_min:roi_y_max, :]
        roi_h, roi_w, _ = roi.shape
        
        # 输入模型 (640, 360)
        inp = cv2.resize(roi, (640, 360))
        inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        inp_t = (inp_t - mean) / std
        
        with torch.no_grad():
            dmap = model(inp_t)[0, 0].cpu().numpy()
            
        current_view_fish = float(np.sum(dmap))
        
        # 门禁带对应的坐标映射
        gx_ratio = gate_x_px / roi_w
        dmap_gx = int(gx_ratio * 640)
        dmap_band_w = int((gate_band_half * 2) / roi_w * 640)
        
        band_left = max(0, dmap_gx - dmap_band_w // 2)
        band_right = min(640, dmap_gx + dmap_band_w // 2)
        
        band_density = np.sum(dmap[:, band_left:band_right])
        # 顺水流速度冲刷转换系数 (根据 60fps 标定)
        # 一条鱼通过 50px 宽度的门禁带平均持续约 1.2 帧
        flux_per_frame = band_density / 1.25
        cum_passage_count += flux_per_frame
        passage_history.append(cum_passage_count)
        
        vis = frame.copy()
        
        # 生成热力图叠加
        dmap_norm = (dmap / (np.max(dmap) + 1e-6) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (roi_w, roi_h))
        
        # 将热力图以半透明方式叠加到水槽区域
        vis_roi = vis[roi_y_min:roi_y_max, :]
        vis[roi_y_min:roi_y_max, :] = cv2.addWeighted(vis_roi, 0.68, heatmap, 0.32, 0)
        
        # 绘制门禁线
        cv2.line(vis, (gate_x_px, roi_y_min), (gate_x_px, roi_y_max), (0, 255, 255), 3)
        cv2.putText(vis, "DENSITY GATE", (gate_x_px - 80, roi_y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        # 绘制 HUD 仪表板
        cv2.rectangle(vis, (30, 30), (600, 170), (20, 20, 20), -1)
        cv2.rectangle(vis, (30, 30), (600, 170), (180, 180, 180), 2)
        cv2.putText(vis, f"CUMULATIVE PASS: {int(cum_passage_count)}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(vis, f"Instant View Density: {current_view_fish:.1f} fish | Band Flux: {flux_per_frame:.2f}",
                    (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(vis, f"Frame: {frame_idx}/{total_frames} ({frame_idx/fps:.1f}s)", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
                    
        writer.write(vis)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  处理进度: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%) | 瞬时鱼群: {current_view_fish:.1f} | 累计通过: {int(cum_passage_count)}...")
            
    cap.release()
    writer.release()
    
    report_file = os.path.splitext(output_path)[0] + "_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== 全量 211 点密度网络 (Density Map) 鱼流分析报告 ===\n")
        f.write(f"视频文件: {video_path}\n")
        f.write(f"总时长: {total_frames/fps:.2f} 秒 ({total_frames} 帧 @ {fps} fps)\n")
        f.write(f"模型权重: runs/density/fish_density_full211.pth (211 极密黄金点训练)\n")
        f.write(f"最终累计通过鱼数: {int(cum_passage_count)} 条\n")
        f.write("------------------------------------\n")
        f.write(f"第一波关键锚点核算:\n")
        f.write(f"  - 第 77 帧累计: {passage_history[77]:.1f} 条 (基准: 1.0)\n")
        f.write(f"  - 第 88 帧累计: {passage_history[88]:.1f} 条 (基准: 3.0)\n")
        f.write(f"  - 第 213 帧累计: {passage_history[213]:.1f} 条 (基准: 73.0)\n")
        
    print(f"\n==========================================")
    print(f"🎉 密度图网络视频全量分析完成！")
    print(f"🐟 最终累计通过鱼数: {int(cum_passage_count)} 条")
    print(f"🎯 第 77 帧: {passage_history[77]:.1f} (目标: 1.0)")
    print(f"🎯 第 88 帧: {passage_history[88]:.1f} (目标: 3.0)")
    print(f"🎯 第 213 帧: {passage_history[213]:.1f} (目标: 73.0)")
    print(f"📹 渲染视频: {output_path}")
    print(f"📄 报告文件: {report_file}")
    print(f"==========================================")

if __name__ == "__main__":
    video = "/Users/qsw/Documents/fishcount/新素材/IMG_2624.MOV"
    output = "/Users/qsw/Documents/fishcount/分析结果/IMG_2624_density_full.mp4"
    run_density_video_pipeline(video, output)
