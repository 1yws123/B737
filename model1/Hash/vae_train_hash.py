import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import argparse
import numpy as np
from schedulers import WarmupCosineScheduler

# 引入你的模型和 Dataset
from vae_model_hash import PointCloudVAE 
from dataset import SDFDataset   

# --- 1. 命令行参数 ---
parser = argparse.ArgumentParser(description='Triplane-VAE Training Script (Hash Encoding Version)')

# 训练控制参数
parser.add_argument('--resume', type=str, default=None, help='checkpoint 路径')
parser.add_argument('--epochs', type=int, default=15000, help='总 Epoch 数')
parser.add_argument('--batch_size', type=int, default=20, help='批次大小')
parser.add_argument('--lr', type=float, default=2e-3, help='最大学习率')
parser.add_argument('--beta_kl', type=float, default=1e-6, help='KL 散度损失权重')

# 模型与数据参数
parser.add_argument('--latent_dim', type=int, default=128, help='潜在空间维度')
parser.add_argument('--plane_res', type=int, default=128, help='Triplane 分辨率')
parser.add_argument('--plane_feat', type=int, default=32, help='Triplane 特征通道数')

# parser.add_argument('--fourier_dim', type=int, default=8, help='(已弃用) 傅里叶特征频率')

# SDF 采样参数
parser.add_argument('--num_points_sdf', type=int, default=16384, help='SDF 采样点总数')
parser.add_argument('--surface_ratio', type=float, default=0.8, help='混合比例')
parser.add_argument('--surface_threshold', type=float, default=0.02, help='表面阈值')

args = parser.parse_args()

# --- 2. 路径与设备设定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PC_ROOT_DIR = "B737_pc/pointcloud" 
SDF_DIR = "B737_sdf/sdf_data"
CHECKPOINT_DIR = "./checkpoint_8(Hash和marchingcubes_npoints1024)" 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 3. 数据与模型加载 ---
NUM_POINTS_UNIFORM = 4000    # 稍微调整以匹配你之前的脚本
NUM_POINTS_CURVATURE = 4000 
NUM_POINTS_IMPORTANCE = 4000 

print(f"==================================================")
print(f"初始化配置 (Hash Encoding 版):")
print(f"  Device: {DEVICE}")
print(f"  Batch Size: {args.batch_size}")
print(f"==================================================")

print("正在初始化 Dataset...")
train_dataset = SDFDataset(
    pc_root_dir=PC_ROOT_DIR, 
    sdf_dir=SDF_DIR,
    num_points_uniform=NUM_POINTS_UNIFORM,
    num_points_curvature=NUM_POINTS_CURVATURE, # 传入
    num_points_importance=NUM_POINTS_IMPORTANCE,
    num_points_sdf=args.num_points_sdf,
    surface_ratio=args.surface_ratio,
    surface_threshold=args.surface_threshold 
)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

print("正在初始化 VAE 模型...")

# [修改点 2] 模型初始化：移除了 num_fourier_freqs 参数
model = PointCloudVAE(
    latent_dim=args.latent_dim,
    plane_resolution=args.plane_res,
    plane_features=args.plane_feat,
    #num_fourier_freqs=args.fourier_dim,
    num_points_uniform=NUM_POINTS_UNIFORM,
    num_points_curvature=NUM_POINTS_CURVATURE, # 传入
    num_points_importance=NUM_POINTS_IMPORTANCE
).to(DEVICE)

# --- 4. 优化器与调度器 ---
# Hash Encoding 的参数通常对 epsilon 比较敏感
# Instant-NGP 使用 eps=1e-15，但在纯 PyTorch float32 下，默认的 1e-8 通常也没问题
# 如果发现 Loss 变成 NaN，可以尝试把 eps 设小一点或大一点
optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8) 

scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 20, total_epochs=args.epochs)
reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')

# --- 5. Checkpoint 加载 (Resume) ---
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
       print(f"=> 正在从 '{args.resume}' 加载 Checkpoint...")
       checkpoint = torch.load(args.resume, map_location=DEVICE)
       
       # 注意：如果之前的 checkpoint 是基于旧模型的，这里可能会报错（参数不匹配）
       # 因为 HashEmbedder 的参数名和 FourierEmbedder 完全不同。
       # 建议从头训练，或者使用 strict=False (不推荐，因为需要训练新的 Embedding)
       try:
           model.load_state_dict(checkpoint['model_state_dict'])
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           start_epoch = checkpoint['epoch']
           print(f"=> Checkpoint 加载完毕。")
       except Exception as e:
           print(f"!! Checkpoint 加载失败 (可能是模型结构变了): {e}")
           print("!! 将从头开始训练...")
    else:
        print(f"!! 警告：找不到 Checkpoint 文件 '{args.resume}'")

# --- 6. 训练循环 ---
print(f"开始训练...")

log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
if not os.path.isfile(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Recon_Loss', 'KL_Loss', 'Total_Loss', 'LR', 'Time(min)'])
        
start_time = time.time()

for epoch in range(start_epoch, args.epochs):
    model.train()
    
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_start_time = time.time()

    for i, batch in enumerate(train_dataloader):
        # 1. 數據搬運
        point_clouds_raw = batch['point_cloud'].to(DEVICE)   # (B, N, 3)
        sdf_points_gt = batch['sdf_points'].to(DEVICE)       # (B, M, 3) 已經是歸一化好的
        sdf_values_gt = batch['sdf_values'].to(DEVICE)
        
        # 獲取歸一化參數，調整維度以進行廣播 (Broadcasting)
        # shift: (B, 3) -> (B, 1, 3)
        #shifts = batch['shift'].to(DEVICE).unsqueeze(1)
        # scale: (B, 1) -> (B, 1, 1)
        #scales = batch['scale'].to(DEVICE).view(-1, 1, 1)
        
        # ============================================================
        # 2. 確定性歸一化 (Deterministic Normalization)
        # 直接復現 box.py 的操作: (pts - centroid) / max_dist
        # ============================================================
        point_clouds_normalized = point_clouds_raw 

        # 3. 前向傳播
        optimizer.zero_grad()
        
        # Encoder 接收嚴格對齊後的點雲
        triplanes, mu, log_var = model(point_clouds_normalized)
        
        # Query SDF (利用 Triplanes 預測空間中 50w 個點的 SDF)
        # 這裡會消耗大量顯存，如果 OOM，請檢查這裡
        sdf_values_pred = model.query_sdf(triplanes, sdf_points_gt)
        
        # 4. Loss 計算
        # 動態 Mask：找出 GT 中真正靠近表面的點 (SDF 值小於閾值)
        # 這比單純依賴 Near 採樣更準確，因為 Vol 採樣也可能隨機採到表面
        surface_mask = torch.abs(sdf_values_gt) < args.surface_threshold
        
        # 加權 Loss: 表面附近的點權重更高 (x3.0)
        # 由於 50w 個點中大部分可能是 Vol (背景)，加強表面權重很重要
        loss_surface = reconstruction_loss_fn(sdf_values_pred[surface_mask], sdf_values_gt[surface_mask])
        loss_non_surface = reconstruction_loss_fn(sdf_values_pred[~surface_mask], sdf_values_gt[~surface_mask])
        
        # 防止 mask 為空導致 nan
        if torch.isnan(loss_surface): loss_surface = 0.0
        if torch.isnan(loss_non_surface): loss_non_surface = 0.0

        reconstruction_loss = loss_surface * 3.0 + loss_non_surface
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        total_loss = reconstruction_loss + args.beta_kl * kl_loss
        
        # 5. 反向傳播
        total_loss.backward()
        optimizer.step()
        
        epoch_recon_loss += reconstruction_loss.item()
        epoch_kl_loss += kl_loss.item()
        
        # (可選) 打印每個 Batch 的進度，因為 50w 點計算較慢
        if (i + 1) % 20== 0:
            print(f"Ep {epoch+1} [{i+1}/{len(train_dataloader)}] Loss: {total_loss.item():.4f} (Recon: {reconstruction_loss.item():.4f})")

    # 每個 Epoch 結束後的處理
    scheduler.step()
    
    avg_recon_loss = epoch_recon_loss / len(train_dataloader)
    avg_kl_loss = epoch_kl_loss / len(train_dataloader)
    current_lr = scheduler.get_last_lr()[0]
    epoch_duration = (time.time() - epoch_start_time) / 60.0 # 分鐘
    
    print(f"Epoch [{epoch+1}/{args.epochs}] Completed in {epoch_duration:.2f} min")
    print(f"  Avg Recon Loss: {avg_recon_loss:.6f}")
    print(f"  Avg KL Loss   : {avg_kl_loss:.4f}")
    print(f"  Current LR    : {current_lr:.8f}")

    # 寫入 CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_recon_loss, avg_kl_loss, avg_recon_loss + args.beta_kl*avg_kl_loss, current_lr, epoch_duration])

    # 保存模型 (每 100 epoch 或其他間隔)
    if (epoch + 1) % 200 == 0 or (epoch + 1) == args.epochs:
        save_path = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # 建議保存 scheduler 狀態
            'args': vars(args) # 保存參數配置以便復現
        }, save_path)
        print(f"Checkpoint 已保存至: {save_path}")

print("全部訓練完成！")