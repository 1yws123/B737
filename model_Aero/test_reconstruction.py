import torch
import numpy as np
import os
import re

# 导入模型定义
from model import PointCloudVAE

# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    CHECKPOINT_PATH = 'checkpoints_stage2_1/stage2_best.pth'
    
    # 指定测试的点云文件
    SINGLE_NPZ_PATH = '/home/yuwenshi/B737/G58_pc_1299/pointcloud/G58_1_pc.npz'
    
    # 【新增】气动数据根目录，用于寻找真值
    AERO_ROOT = '/home/yuwenshi/B737/G58_aero_1299/G58_aero_1299'

    # 模型结构参数 (必须与训练一致)
    LATENT_DIM = 128  
    PLANE_RES = 128
    PLANE_FEAT = 32
    FOURIER_DIM = 8
    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000

# ==========================================
# 2. 辅助工具函数
# ==========================================
def get_ground_truth(npz_path, aero_root):
    """
    根据点云文件名寻找对应的气动真值文件 (.polar)
    """
    file_id = os.path.basename(npz_path).replace('_pc.npz', '')
    # 匹配 G58_8 这种格式
    polar_path = os.path.join(aero_root, file_id, f"{file_id}_VSPGeom.polar")
    
    if not os.path.exists(polar_path):
        return None
    
    try:
        with open(polar_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('0.000'): # 假设 Alpha=0 的数据
                    parts = re.split(r'\s+', line.strip())
                    cl = float(parts[4])
                    cd = float(parts[9])
                    return cl, cd
    except:
        return None
    return None

def resample_points(points, target_num):
    num_curr = points.shape[0]
    if num_curr == 0: return np.zeros((target_num, 3), dtype=np.float32)
    choice = np.random.choice(num_curr, target_num, replace=(num_curr < target_num))
    return points[choice]

# ==========================================
# 3. 执行预测
# ==========================================
def predict_with_error_analysis():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化并加载模型
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM, plane_resolution=cfg.PLANE_RES,
        plane_features=cfg.PLANE_FEAT, num_fourier_freqs=cfg.FOURIER_DIM,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM,
        num_points_curvature=cfg.NUM_POINTS_CURVATURE,
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(device)

    state_dict = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    # 2. 加载点云
    data = np.load(cfg.SINGLE_NPZ_PATH)
    u_pts = resample_points(data['uniform'], cfg.NUM_POINTS_UNIFORM)
    c_pts = resample_points(data['curvature'], cfg.NUM_POINTS_CURVATURE)
    i_pts = resample_points(data['importance'], cfg.NUM_POINTS_IMPORTANCE)
    pc_tensor = torch.from_numpy(np.concatenate([u_pts, c_pts, i_pts], axis=0)).float().unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        _, mu, _, aero_pred = model(pc_tensor, query_points=None)
        pred_cl, pred_cd = aero_pred[0, 0].item(), aero_pred[0, 1].item()

    # 4. 获取真值并计算误差
    gt = get_ground_truth(cfg.SINGLE_NPZ_PATH, cfg.AERO_ROOT)
    
    # 5. 格式化输出
    print("\n" + "展开分析: " + os.path.basename(cfg.SINGLE_NPZ_PATH))
    print("="*60)
    print(f"{'参数 (Alpha=0)':<15} | {'预测值 (Pred)':<12} | {'真值 (GT)':<12} | {'相对误差 (%)':<12}")
    print("-" * 60)
    
    if gt:
        gt_cl, gt_cd = gt
        cl_err = abs(pred_cl - gt_cl) / (abs(gt_cl) + 1e-8) * 100
        cd_err = abs(pred_cd - gt_cd) / (abs(gt_cd) + 1e-8) * 100
        
        print(f"{'CL (升力系数)':<13} | {pred_cl:12.6f} | {gt_cl:12.6f} | {cl_err:11.2f}%")
        print(f"{'CD (阻力系数)':<13} | {pred_cd:12.6f} | {gt_cd:12.6f} | {cd_err:11.2f}%")
        
        # 升阻比单独算
        ld_pred = pred_cl / (pred_cd + 1e-9)
        ld_gt = gt_cl / (gt_cd + 1e-9)
        ld_err = abs(ld_pred - ld_gt) / (abs(ld_gt) + 1e-8) * 100
        print(f"{'L/D (升阻比)':<13} | {ld_pred:12.2f} | {ld_gt:12.2f} | {ld_err:11.2f}%")
    else:
        print(f"CL: {pred_cl:.6f} (未找到真值文件)")
        print(f"CD: {pred_cd:.6f} (未找到真值文件)")
    
    print("="*60)

if __name__ == "__main__":
    predict_with_error_analysis()