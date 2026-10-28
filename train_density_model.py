import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from density_model import FishDensityNet, FishPointDataset

def train_density():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 启动密集点密度图网络训练，运算设备: {device}")
    
    img_dir = "/Users/qsw/Documents/fishcount/待标注_急流黑鱼10张"
    pt_dir = "/Users/qsw/Documents/fishcount/待标注_急流黑鱼10张/points"
    
    dataset = FishPointDataset(img_dir, pt_dir, target_size=(640, 360), sigma=16, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = FishDensityNet().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_dmap = nn.SmoothL1Loss(beta=0.01)
    
    out_dir = "/Users/qsw/Documents/fishcount/runs/density"
    os.makedirs(out_dir, exist_ok=True)
    best_weight_path = os.path.join(out_dir, "fish_density_best.pth")
    
    best_loss = 999999.0
    epochs = 40
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count_err = 0.0
        batches = 0
        
        for imgs, dmaps, true_counts in loader:
            imgs = imgs.to(device)
            dmaps = dmaps.to(device)
            
            optimizer.zero_grad()
            pred_dmaps = model(imgs)
            
            loss_pixel = criterion_dmap(pred_dmaps, dmaps) * 100.0
            
            # 整体计数守恒约束
            pred_counts = torch.sum(pred_dmaps, dim=(1, 2, 3))
            gt_counts = torch.sum(dmaps, dim=(1, 2, 3))
            loss_count = torch.mean(torch.abs(pred_counts - gt_counts))
            
            loss = loss_pixel + 0.5 * loss_count
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_count_err += loss_count.item()
            batches += 1
            
        avg_loss = total_loss / batches
        avg_count_err = total_count_err / batches
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg_loss:.4f} | Avg Count Error: {avg_count_err:.2f} fish")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_weight_path)
            
    print(f"🎉 训练完成！最优模型已保存至: {best_weight_path}")

if __name__ == "__main__":
    train_density()
