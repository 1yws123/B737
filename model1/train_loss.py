import csv
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse
import numpy as np
from schedulers import WarmupCosineScheduler  # 確保你有這個文件

# ===============================================================
# 1. 核心 Loss 辅助函数 (参考论文实现)
# ===============================================================

def compute_tv_loss(triplanes):
    """全变分正则化：让 Triplane 特征图在空间上更平滑，减少高频噪点"""
    loss = 0
    for plane in triplanes:
        # plane shape: [B, C, H, W]
        tv_h = torch.pow(plane[:, :, 1:, :] - plane[:, :, :-1, :], 2).mean()
        tv_w = torch.pow(plane[:, :, :, 1:] - plane[:, :, :, :-1], 2).mean()
        loss += (tv_h + tv_w)
    return loss / 3.0

def compute_triplane_l2_loss(triplanes):
    """Triplane L2 正则化：约束特征值范围，防止数值爆炸，增强 VAE 稳定性"""
    loss = 0
    for plane in triplanes:
        loss += torch.mean(plane ** 2)
    return loss / 3.0

def compute_edr_loss(model_module, triplanes, device, num_samples=1024):
    """显式密度正则化 (EDR)：确保空间 SDF 场的连续性，消除空中浮动伪影"""
    batch_size = triplanes[0].shape[0]
    # 在 [-1, 1] 空间内随机采样点
    pts = torch.rand((batch_size, num_samples, 3), device=device) * 2 - 1
    
    # 施加微小扰动
    eps = 0.005
    pts_perturbed = pts + torch.randn_like(pts) * eps
    pts_perturbed = torch.clamp(pts_perturbed, -1, 1)
    
    # 通过 Triplane 查询预测值
    out1 = model_module.query_sdf(triplanes, pts)
    out2 = model_module.query_sdf(triplanes, pts_perturbed)
    
    return F.mse_loss(out1, out2)

# ===============================================================
# 2. 训练配置与参数
# ===============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

parser = argparse.ArgumentParser(description='Triplane-VAE Training with Paper Regularizations')

# 保持你原有的参数
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--beta_kl', type=float, default=1e-6)

# 新增正则化权重参数 (建议初值)
parser.add_argument('--lambda_tv', type=float, default=0.01, help='TV 平滑权重')
parser.add_argument('--lambda_l2', type=float, default=1e-4, help='Triplane L2 权重')
parser.add_argument('--lambda_edr', type=float, default=0.1, help='EDR 权重')

# 模型与数据参数
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--plane_res', type=int, default=128)
parser.add_argument('--plane_feat', type=int, default=32)
parser.add_argument('--fourier_dim', type=int, default=8)
parser.add_argument('--num_points_sdf', type=int, default=100000)
parser.add_argument('--surface_ratio', type=float, default=0.8)
parser.add_argument('--surface_threshold', type=float, default=0.02)

args = parser.parse_args()

# 设备设定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PC_ROOT_DIR = "B737_pc/pointcloud" 
SDF_DIR = "B737_sdf/sdf_data"
CHECKPOINT_DIR = "./checkpoint_7(Fourier和marchingcubes_含Loss)"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 导入模型和数据
from model import PointCloudVAE
from dataset import SDFDataset 

# --- 初始化 Dataset ---
NUM_POINTS_UNIFORM = 4000 
NUM_POINTS_CURVATURE = 4000 
NUM_POINTS_IMPORTANCE = 4000 

train_dataset = SDFDataset(
    pc_root_dir=PC_ROOT_DIR, sdf_dir=SDF_DIR,
    num_points_uniform=NUM_POINTS_UNIFORM,
    num_points_curvature=NUM_POINTS_CURVATURE,
    num_points_importance=NUM_POINTS_IMPORTANCE,
    num_points_sdf=args.num_points_sdf,
    surface_ratio=args.surface_ratio,
    surface_threshold=args.surface_threshold 
)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

# --- 初始化 Model ---
model = PointCloudVAE(
    latent_dim=args.latent_dim, plane_resolution=args.plane_res,
    plane_features=args.plane_feat, num_fourier_freqs=args.fourier_dim,
    num_points_uniform=NUM_POINTS_UNIFORM,
    num_points_curvature=NUM_POINTS_CURVATURE,
    num_points_importance=NUM_POINTS_IMPORTANCE
).to(DEVICE)

# 多卡并行处理
model_module = model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model_module = model.module

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 20, total_epochs=args.epochs)
reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')

# --- 断点续传逻辑 ---
start_epoch = 0
if args.resume and os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"=> 加载完成，从 Epoch {start_epoch + 1} 继续")

# --- 日志记录 ---
log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
if not os.path.isfile(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Recon_Loss', 'KL_Loss', 'TV_Loss', 'EDR_Loss', 'Total_Loss', 'LR', 'Time(min)'])

# ===============================================================
# 3. 训练循环
# ===============================================================

print(f"开始训练，设备: {DEVICE}...")

for epoch in range(start_epoch, args.epochs):
    model.train()
    epoch_metrics = {k: 0.0 for k in ['recon', 'kl', 'tv', 'edr', 'total']}
    epoch_start_time = time.time()

    for i, batch in enumerate(train_dataloader):
        # 1. 数据准备
        point_clouds_raw = batch['point_cloud'].to(DEVICE)
        sdf_points_gt = batch['sdf_points'].to(DEVICE)
        sdf_values_gt = batch['sdf_values'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # 2. 前向传播
        triplanes, mu, log_var = model(point_clouds_raw)
        
        # 3. 计算查询 SDF 预测
        sdf_values_pred = model_module.query_sdf(triplanes, sdf_points_gt)
        
        # 4. 计算各项 Loss
        # (A) 重建损失 (带表面加权)
        surface_mask = torch.abs(sdf_values_gt) < args.surface_threshold
        loss_surface = reconstruction_loss_fn(sdf_values_pred[surface_mask], sdf_values_gt[surface_mask])
        loss_non_surface = reconstruction_loss_fn(sdf_values_pred[~surface_mask], sdf_values_gt[~surface_mask])
        
        # 处理空 Mask 情况
        loss_surface = loss_surface if not torch.isnan(loss_surface) else torch.tensor(0.0).to(DEVICE)
        loss_non_surface = loss_non_surface if not torch.isnan(loss_non_surface) else torch.tensor(0.0).to(DEVICE)
        
        reconstruction_loss = loss_surface * 3.0 + loss_non_surface
        
        # (B) VAE KL Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # (C) 新增正则化项
        tv_loss = compute_tv_loss(triplanes)
        l2_tri_loss = compute_triplane_l2_loss(triplanes)
        edr_loss = compute_edr_loss(model_module, triplanes, DEVICE)
        
        # (D) 总损失汇总
        total_loss = reconstruction_loss + \
                     args.beta_kl * kl_loss + \
                     args.lambda_tv * tv_loss + \
                     args.lambda_l2 * l2_tri_loss + \
                     args.lambda_edr * edr_loss
        
        # 5. 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 累计指标
        epoch_metrics['total'] += total_loss.item()
        epoch_metrics['recon'] += reconstruction_loss.item()
        epoch_metrics['kl'] += kl_loss.item()
        epoch_metrics['tv'] += tv_loss.item()
        epoch_metrics['edr'] += edr_loss.item()

        if (i + 1) % 20 == 0:
            print(f"Ep {epoch+1} [{i+1}/{len(train_dataloader)}] Total: {total_loss.item():.4f} | Recon: {reconstruction_loss.item():.4f} | TV: {tv_loss.item():.4f}")

    # Epoch 结束处理
    scheduler.step()
    num_batches = len(train_dataloader)
    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
    current_lr = scheduler.get_last_lr()[0]
    epoch_duration = (time.time() - epoch_start_time) / 60.0
    
    print(f"Epoch [{epoch+1}/{args.epochs}] Avg Recon: {avg_metrics['recon']:.6f} | TV: {avg_metrics['tv']:.6f} | Time: {epoch_duration:.2f} min")

    # 写入 CSV 日志
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1, 
            avg_metrics['recon'], 
            avg_metrics['kl'], 
            avg_metrics['tv'], 
            avg_metrics['edr'], 
            avg_metrics['total'], 
            current_lr, 
            epoch_duration
        ])

    # 保存模型
    if (epoch + 1) % 200 == 0 or (epoch + 1) == args.epochs:
        save_path = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args)
        }, save_path)
        print(f"✅ Checkpoint 已保存: {save_path}")

print("训练全部完成！")