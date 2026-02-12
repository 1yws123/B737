import os
import time
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from schedulers import WarmupCosineScheduler 

from model import PointCloudVAE
from dataset import SDFDataset

def get_args():
    parser = argparse.ArgumentParser(description='Stage 2: Frozen Encoder Aero Training')
    
    # --- 路径配置 ---
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/G58_sdf_1299/sdf_data')
    parser.add_argument('--stage1_ckpt', type=str, default='/home/yuwenshi/B737/checkpoint_all_1/vae_epoch_8400.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints_stage2_3')

    parser.add_argument('--surface_ratio', type=float, default=0.8, help='SDF 采样中表面点的比例')
    parser.add_argument('--surface_threshold', type=float, default=0.02, help='表面点判定的阈值')
    parser.add_argument('--num_points_sdf', type=int, default=250000, help='SDF 採樣點總數 (Vol + Near)')

    parser.add_argument('--latent_dim', type=int, default=128, help='潛在空間維度')
    parser.add_argument('--plane_res', type=int, default=128, help='Triplane 特徵平面的分辨率 (64 或 128 可提升細節)')
    parser.add_argument('--plane_feat', type=int, default=32, help='Triplane 特徵通道數')
    parser.add_argument('--fourier_dim', type=int, default=8, help='傅立葉特徵頻率數量')
    parser.add_argument('--val_ratio', type=float, default=0.2, help="验证集比例 (0.2 = 20%)")

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=60, help='因为冻结了大部分参数,显存占用小,Batch可调大')
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # --- 论文公式对应的系数 ---
    # CL 约 0.5, CD 约 0.02。CD 的 MSE 会非常小 (1e-4级别)。
    # alpha=1.0, beta=100.0 (让 CD 的梯度放大100倍)
    #parser.add_argument('--alpha', type=float, default=1.0, help='Weight for CL')
    #parser.add_argument('--beta', type=float, default=100.0, help='Weight for CD')

    return parser.parse_args()

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    
    with torch.no_grad():
        for data in val_loader:
            pc = data['point_cloud'].to(device)
            # GT Shape: [B, 2] -> 我们只取第一列 CL -> [B, 1]
            gt_aero = data['aero_label'].to(device).float()
            gt_cl = gt_aero[:, 0].unsqueeze(1) 

            # Forward
            _, _, _, aero_pred = model(pc) # aero_pred: [B, 1]
            
            # Loss
            loss = criterion(aero_pred, gt_cl)
            val_loss += loss.item()
            
            # MAE
            val_mae += torch.mean(torch.abs(aero_pred - gt_cl)).item()
            
    avg_loss = val_loss / len(val_loader)
    avg_mae = val_mae / len(val_loader)
    return avg_loss, avg_mae

def main():
    args = get_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
        
    print(f"开始 CL 单任务训练 | Epochs: {args.epochs} | LR: {args.lr}")

    # 1. 数据集
    dataset = SDFDataset(
        surface_ratio=args.surface_ratio,
        surface_threshold=args.surface_threshold,
        pc_root_dir=args.pc_root,
        aero_root_dir=args.aero_root,
        sdf_dir=args.sdf_dir,
        num_points_uniform=4000, num_points_curvature=4000, num_points_importance=4000
    )
    # 划分 训练集 / 验证集
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"数据集: 总数 {len(dataset)} | 训练集 {train_size} | 验证集 {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 模型加载
    model = PointCloudVAE(
       latent_dim=args.latent_dim,
       plane_resolution=args.plane_res,
       plane_features=args.plane_feat,
       num_fourier_freqs=args.fourier_dim,
       num_points_uniform=4000,
       num_points_curvature=4000, # 传入
       num_points_importance=4000
).to(device)
    
    if os.path.exists(args.stage1_ckpt):
        print(f"📥 加载 Stage 1 权重: {args.stage1_ckpt}")
        ckpt = torch.load(args.stage1_ckpt, map_location=device,weights_only=False)
        model.load_state_dict(ckpt, strict=False)
    else:
        raise FileNotFoundError("必须提供 Stage 1 权重才能进行冻结训练！")

    # ========================================================================
    # 3. 核心步骤：冻结几何参数 (Freeze Parameters)
    # ========================================================================
    # 冻结 Encoder
    for param in model.encoder.parameters(): param.requires_grad = False

    # 冻结 Triplane Decoder
    for param in model.decoder.parameters(): param.requires_grad = False
    
    # 确保 Aero Decoder 可训练
    for param in model.aero_decoder.parameters(): param.requires_grad = True
        
    print("已冻结 Encoder 和 Decoder 参数，仅训练 Aero Branch")

    # 4. 优化器：只传入 aero_decoder 的参数
    optimizer = optim.Adam(model.aero_decoder.parameters(), lr=args.lr)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 20, total_epochs=args.epochs)
    criterion = nn.MSELoss()
    
    # 5. 记录日志
    log_file = os.path.join(args.save_dir, 'train_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_MAE', 'LR'])

    # 6. 训练循环
    best_val_mae = float('inf')
    
    for epoch in range(args.epochs):
        model.train() # 训练模式 (对于冻结层，Dropout行为取决于实现，通常建议BN eval模式)
        model.aero_decoder.train() # 确保 decoder 是 train
        epoch_loss = 0
        
        for i, data in enumerate(train_loader):
            pc = data['point_cloud'].to(device)
            # 标签处理: 只取 CL
            gt_aero = data['aero_label'].to(device).float()
            gt_cl = gt_aero[:, 0].unsqueeze(1) # [B, 1]

            optimizer.zero_grad()
            
            # Forward
            _, _, _, aero_pred = model(pc)
            
            loss = criterion(aero_pred, gt_cl)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # === 验证 ===
        # 每 200 轮(或者是最后几轮进行验证)
        if (epoch + 1) % 1 == 0 :
            val_loss, val_mae = validate(model, val_loader, device, criterion)
            
            print(f"Epoch {epoch+1:4d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.6f}")
            
            # 记录日志
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch+1, avg_train_loss, val_loss, val_mae, current_lr])
            
            # 保存最佳模型
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_cl_model.pth'))
                print(f"Best Model Saved (MAE: {best_val_mae:.4f})")
        else:
            # 简略打印
            print(f"\rEpoch {epoch+1:4d} | Train Loss: {avg_train_loss:.6f}", end="")

    print(f"\n训练结束! 最佳验证集 MAE: {best_val_mae:.4f}")

if __name__ == "__main__":
    main()