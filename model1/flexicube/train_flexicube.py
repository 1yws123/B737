import csv
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import argparse
import numpy as np

# 导入必要模块
from model_flexicube import PointCloudVAE
from schedulers import WarmupCosineScheduler
from dataset import SDFDataset
from flexicubes import FlexiCubes

# 代码最顶部添加（优先执行）
# 1. 指定空闲GPU（比如GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 2. 显存优化配置
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# 修改DEVICE初始化逻辑
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

# --- 1. 命令行參數 ---
parser = argparse.ArgumentParser(description='Triplane-VAE FlexiCubes Training')

# 基础参数
parser.add_argument('--resume', type=str, default=None, help='checkpoint 檔案路徑')
parser.add_argument('--epochs', type=int, default=8000, help='總訓練 Epoch 數')
parser.add_argument('--batch_size', type=int, default=25, help='批次大小')
parser.add_argument('--lr', type=float, default=2e-3, help='最大學習率')
parser.add_argument('--beta_kl', type=float, default=1e-6, help='KL 散度損失的權重')

# FlexiCubes 特定参数
parser.add_argument('--w_dev', type=float, default=0.01, help='FlexiCubes L_dev 正则化损失权重')
parser.add_argument('--flex_res', type=int, default=32, help='训练时用于计算 FlexiCubes Loss 的网格分辨率 (32或64)')

# 模型参数
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--plane_res', type=int, default=128)
parser.add_argument('--plane_feat', type=int, default=32)
parser.add_argument('--fourier_dim', type=int, default=8)

# 数据参数
parser.add_argument('--num_points_sdf', type=int, default=16384)
parser.add_argument('--surface_ratio', type=float, default=0.8)
parser.add_argument('--surface_threshold', type=float, default=0.02)

args = parser.parse_args()

# --- 2. 路徑與設備設定 ---

PC_ROOT_DIR = "B737_pc/pointcloud" 
SDF_DIR = "B737_sdf/sdf_data"
CHECKPOINT_DIR = "./checkpoint_4(Fourier和flexicubes)"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 3. 辅助函数：构建网格拓扑 (User Provided) ---
# 这个函数生成的顶点范围是 [-1, 1]，完美适配 SDF
def construct_grid_topology(res, device):
    grid_vals = torch.linspace(-1.0, 1.0, res + 1, device=device)
    grid_verts = torch.stack(torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij'), dim=-1)
    grid_verts = grid_verts.reshape(-1, 3)

    i, j, k = torch.meshgrid(
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        torch.arange(res, device=device), indexing='ij')
    
    base_indices = k + (res + 1) * j + (res + 1) * (res + 1) * i
    base_indices = base_indices.reshape(-1)

    offsets = torch.tensor([
        0, 1, (res + 1), (res + 1) + 1,
        (res + 1)**2, (res + 1)**2 + 1, (res + 1)**2 + (res + 1), (res + 1)**2 + (res + 1) + 1
    ], device=device)
    
    cubes = base_indices[:, None] + offsets[None, :]
    return grid_verts, cubes

# --- 4. 數據與模型加載 ---
NUM_POINTS_UNIFORM = 4000    # 稍微调整以匹配你之前的脚本
NUM_POINTS_CURVATURE = 4000 
NUM_POINTS_IMPORTANCE = 4000 

print(f"==================================================")
print(f"初始化配置 (FlexiCubes):")
print(f"  Flex Grid Res: {args.flex_res}")
print(f"  L_dev Weight: {args.w_dev}")
print(f"  Device: {DEVICE}")
print(f"  SDF Points: {args.num_points_sdf}")
print(f"  Batch Size: {args.batch_size}")
print(f"  Surface Ratio (Near/Vol): {args.surface_ratio}")
print(f"==================================================")

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

model = PointCloudVAE(
    latent_dim=args.latent_dim,
    plane_resolution=args.plane_res,
    plane_features=args.plane_feat,
    num_fourier_freqs=args.fourier_dim,
    num_points_uniform=NUM_POINTS_UNIFORM,
    num_points_curvature=NUM_POINTS_CURVATURE, # 传入
    num_points_importance=NUM_POINTS_IMPORTANCE
).to(DEVICE)

# --- 5. 初始化 FlexiCubes 和 Fixed Grid ---
fc = FlexiCubes(device=DEVICE)

# 使用你的函数构建网格
voxel_verts, voxel_cubes = construct_grid_topology(args.flex_res, DEVICE)
voxel_verts = voxel_verts.float()
voxel_cubes = voxel_cubes.long()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 20, total_epochs=args.epochs)
reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')

# --- 6. Checkpoint 加載 ---
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
       checkpoint = torch.load(args.resume, map_location=DEVICE)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       start_epoch = checkpoint['epoch']
       print(f"=> 加載 Checkpoint 成功。")
    else:
        print(f"!! 警告：找不到 Checkpoint。")

# --- 7. 訓練循環 ---
print(f"開始訓練...")
log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
if not os.path.isfile(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Recon_Loss', 'KL_Loss', 'Dev_Loss', 'Total_Loss', 'LR', 'Time(min)'])
        
start_time = time.time()

for epoch in range(start_epoch, args.epochs):
    model.train()
    
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_dev_loss = 0.0
    epoch_start_time = time.time()

    for i, batch in enumerate(train_dataloader):
        point_clouds_raw = batch['point_cloud'].to(DEVICE)
        sdf_points_gt = batch['sdf_points'].to(DEVICE)
        sdf_values_gt = batch['sdf_values'].to(DEVICE)
        
        point_clouds_normalized = point_clouds_raw 

        optimizer.zero_grad()
        
        # 1. Forward
        # 这里之前报错是因为 model 返回的 flex_weights 是 None
        # 现在修改 model 后应该返回 (B, 21) 的 tensor
        triplanes, mu, log_var, flex_weights = model(point_clouds_normalized)
        
        # 2. SDF Recon Loss
        sdf_values_pred = model.query_sdf(triplanes, sdf_points_gt)
        
        surface_mask = torch.abs(sdf_values_gt) < args.surface_threshold
        loss_surface = reconstruction_loss_fn(sdf_values_pred[surface_mask], sdf_values_gt[surface_mask])
        loss_non_surface = reconstruction_loss_fn(sdf_values_pred[~surface_mask], sdf_values_gt[~surface_mask])
        
        if torch.isnan(loss_surface): loss_surface = 0.0
        if torch.isnan(loss_non_surface): loss_non_surface = 0.0

        reconstruction_loss = loss_surface * 3.0 + loss_non_surface
        
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # 3. FlexiCubes Loss (L_dev)
        dev_loss_batch = 0.0
        B = point_clouds_normalized.shape[0]
        
        # 扩展 voxel_verts 以进行批量查询
        #batch_voxel_verts = voxel_verts.unsqueeze(0).expand(B, -1, -1) # (B, N, 3)
        
        # 查询 Grid 上的 SDF
        #grid_sdf_pred = model.query_sdf(triplanes, batch_voxel_verts).squeeze(-1) # (B, N)
        
        for b in range(B):
            w = flex_weights[b] # (21,)
            triplane_sample = (
                triplanes[0][b:b+1], # 取出 XY 平面的第 b 個樣本
                triplanes[1][b:b+1], # 取出 YZ 平面的第 b 個樣本
                triplanes[2][b:b+1]  # 取出 XZ 平面的第 b 個樣本
            )
            verts_sample = voxel_verts.unsqueeze(0)
            s_n = model.query_sdf(triplane_sample, verts_sample).squeeze(-1).squeeze(0)
            # 扩展参数到每个 cube
            n_cubes = voxel_cubes.shape[0]
            beta = w[:12].unsqueeze(0).expand(n_cubes, -1)
            alpha = w[12:20].unsqueeze(0).expand(n_cubes, -1)
            gamma = w[20].unsqueeze(0).expand(n_cubes)
            
            try:
                # 调用 FlexiCubes
                # x_nx3: 使用你函数生成的 [-1, 1] 顶点
                # s_n: 预测的 SDF 值
                # beta, alpha, gamma: 模型预测的权重
                _, _, l_dev = fc(
                    x_nx3=voxel_verts, 
                    s_n=s_n, # 這裡改用上面計算的單個 s_n
                    cube_fx8=voxel_cubes,
                    res=args.flex_res,
                    beta_fx12=beta,
                    alpha_fx8=alpha,
                    gamma_f=gamma,
                    training=True,
                    grad_func=None
                )
                
                if l_dev.numel() > 0:
                    dev_loss_batch += l_dev.mean()
            except Exception as e:
                pass

        dev_loss = dev_loss_batch / B
        
        # 4. Total Loss
        total_loss = reconstruction_loss + args.beta_kl * kl_loss + args.w_dev * dev_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_recon_loss += reconstruction_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_dev_loss += dev_loss.item()
        
        if (i + 1) % 20 == 0:
            print(f"Ep {epoch+1} [{i+1}/{len(train_dataloader)}] "
                  f"Total: {total_loss.item():.4f} | "
                  f"Recon: {reconstruction_loss.item():.4f} | "
                  f"Dev: {dev_loss.item():.4f}")

    scheduler.step()
    
    avg_recon = epoch_recon_loss / len(train_dataloader)
    avg_kl = epoch_kl_loss / len(train_dataloader)
    avg_dev = epoch_dev_loss / len(train_dataloader)
    current_lr = scheduler.get_last_lr()[0]
    epoch_duration = (time.time() - epoch_start_time) / 60.0
    
    print(f"Epoch [{epoch+1}/{args.epochs}] {epoch_duration:.2f} min")
    print(f"  Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}, Dev: {avg_dev:.6f}")

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_recon, avg_kl, avg_dev, avg_recon + args.beta_kl*avg_kl + args.w_dev*avg_dev, current_lr, epoch_duration])

    if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
        save_path = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args)
        }, save_path)
        print(f"Checkpoint Saved: {save_path}")

print("全部訓練完成！")