import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class FishDensityNet(nn.Module):
    """
    轻量级高抗遮挡纯卷积密度网络 (纯离线，无需外网下载预训练)
    采用多层空洞卷积 (Dilated Convolutions) 快速构建超大感受野
    """
    def __init__(self):
        super(FishDensityNet, self).__init__()
        
        # 特征提取前端 (快速下采样至 1/4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度空洞特征融合层 (Dilation=2, 4, 8 捕获叠罗汉不同尺度)
        self.dilated_blocks = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 高分辨率密度图恢复解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU() # 密度绝对非负
        )
        
    def forward(self, x):
        feat = self.encoder(x)
        feat_dilated = self.dilated_blocks(feat)
        dmap = self.decoder(feat_dilated)
        return dmap

class FishPointDataset(Dataset):
    def __init__(self, img_dir, pt_dir, target_size=(640, 360), sigma=18, augment=True):
        self.img_dir = img_dir
        self.pt_dir = pt_dir
        self.target_size = target_size # (w, h)
        self.sigma = sigma
        self.augment = augment
        
        self.pt_files = sorted(glob.glob(os.path.join(pt_dir, "*_pts.txt")))
        self.samples = []
        for pf in self.pt_files:
            base = os.path.basename(pf).replace("_pts.txt", "")
            img_path = os.path.join(img_dir, f"{base}.jpg")
            if os.path.exists(img_path):
                self.samples.append((img_path, pf))
                
    def __len__(self):
        # 数据集样本较少时，虚拟复制扩展 epoch 长度
        return len(self.samples) * 8 if self.augment else len(self.samples)
        
    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        img_path, pt_path = self.samples[real_idx]
        
        img = cv2.imread(img_path)
        orig_h, orig_w, _ = img.shape
        
        # 水槽有效 ROI
        roi = img[80:980, :]
        roi_h, roi_w, _ = roi.shape
        
        pts = []
        with open(pt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    px, py = map(float, parts[:2])
                    # 转换到 roi 坐标
                    pts.append((px, py - 80))
                    
        tw, th = self.target_size
        roi_resized = cv2.resize(roi, (tw, th))
        
        # 尺度变换
        sx = tw / roi_w
        sy = th / roi_h
        
        # 生成高斯密度图
        dmap = np.zeros((th, tw), dtype=np.float32)
        valid_pts = []
        for px, py in pts:
            nx = px * sx
            ny = py * sy
            if 0 <= nx < tw and 0 <= ny < th:
                valid_pts.append((nx, ny))
                dmap[int(ny), int(nx)] += 1.0
                
        if len(valid_pts) > 0:
            dmap_smooth = gaussian_filter(dmap, sigma=self.sigma)
            # 保证总积分为精确点数
            dmap_smooth = dmap_smooth / (np.sum(dmap_smooth) + 1e-7) * len(valid_pts)
        else:
            dmap_smooth = dmap
            
        # 简单随机数据增强
        if self.augment and np.random.rand() > 0.5:
            roi_resized = cv2.flip(roi_resized, 0) # 垂直翻转 (水流依旧从右向左)
            dmap_smooth = cv2.flip(dmap_smooth, 0)
            
        # 归一化输入图像
        img_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        dmap_tensor = torch.from_numpy(dmap_smooth).unsqueeze(0).float()
        
        return img_tensor, dmap_tensor, len(valid_pts)
