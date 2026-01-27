import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import binary_dilation
from skimage import measure
import time
from collections import OrderedDict

from model import PointCloudVAE

# 强制使用 GPU，因为高分辨率推理在 CPU 上极慢
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 配置参数
# =============================================================================
class Config:
    # 1. 权重路径：确保指向你最新的 checkpoint
    MODEL_PATH = r"checkpoint_7(Fourier和marchingcubes_含Loss)/vae_epoch_10000.pth" 
    INPUT_PC_PATH = r"B737_pc/pointcloud/G58_8_pc.npz"
    # 增加文件名标识，防止混淆
    OUTPUT_MESH_PATH = r"reconstruct/reconstruct_G58_8_7_10000epoch.obj"

    # 2. 模型结构参数 (必须与 train_loss.py 严格一致)
    LATENT_DIM = 128
    PLANE_RES = 128  # 训练时设定的分辨率
    PLANE_FEAT = 32
    FOURIER_DIM = 8

    # 3. 三路点云数量 (必须与训练设置一致)
    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000

    # 4. 重建参数
    # 因为训练了 100k 点，建议分辨率到 512 以获得更平滑的表面
    OCTREE_RESOLUTIONS = [32, 64, 128, 256, 512] 
    SDF_QUERY_BATCH_SIZE = 65536 
    MARCHING_CUBES_LEVEL = 0.0
    BBOX_MIN, BBOX_MAX = -1.0, 1.0

# =============================================================================
# 辅助函数
# =============================================================================
def resample_points(points, target_num):
    num_curr = points.shape[0]
    if num_curr == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    choice = np.random.choice(num_curr, target_num, replace=(num_curr < target_num))
    return points[choice]

def octree_refined_sdf(model, triplanes, resolutions, device, bbox_size, sdf_query_batch_size):
    coarsest_res = resolutions[0]
    grid_min, grid_max = -bbox_size / 2, bbox_size / 2

    grid_vals = torch.linspace(grid_min, grid_max, coarsest_res+1, device=device)
    grid_pts = torch.stack(torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij'), dim=-1)
    
    with torch.no_grad():
        sdf_coarse = model.query_sdf(triplanes, grid_pts.reshape(1, -1, 3))
    grid_sdf = sdf_coarse.reshape(coarsest_res+1, coarsest_res+1, coarsest_res+1)

    for i in range(len(resolutions)-1):
        cur_res, next_res = resolutions[i], resolutions[i+1]
        step = bbox_size / next_res
        
        # 寻找表面附近的体素
        threshold = bbox_size / cur_res 
        mask = (grid_sdf.abs() < threshold)
        mask_np = binary_dilation(mask.cpu().numpy(), iterations=1)
        mask = torch.from_numpy(mask_np).to(device)
        
        inds = torch.nonzero(mask, as_tuple=False)
        offset = torch.tensor([[dx,dy,dz] for dx in [0,1] for dy in [0,1] for dz in [0,1]], device=device)
        inds = (inds.unsqueeze(1) * 2 + offset.unsqueeze(0)).reshape(-1, 3)
        
        inds = inds[(inds >= 0).all(dim=1) & (inds <= next_res).all(dim=1)]
        inds = torch.unique(inds, dim=0)
        
        query_pts = inds.float() * step + grid_min
        next_logits = torch.full((next_res+1, next_res+1, next_res+1), 1.0, dtype=grid_sdf.dtype, device=device)
        
        for j in range(0, len(inds), sdf_query_batch_size):
            batch_inds = inds[j : j + sdf_query_batch_size]
            batch_pts = query_pts[j : j + sdf_query_batch_size].unsqueeze(0)
            sdf_vals = model.query_sdf(triplanes, batch_pts).squeeze(0).squeeze(-1)
            next_logits[batch_inds[:,0], batch_inds[:,1], batch_inds[:,2]] = sdf_vals
            
        factor = next_res // cur_res
        next_logits[::factor, ::factor, ::factor] = grid_sdf
        grid_sdf = next_logits
        
    return grid_sdf

# =============================================================================
# 主重建流程
# =============================================================================
def reconstruct_mesh_with_sdf(cfg: Config):
    print(f"使用的设备: {DEVICE}")

    # 1. 初始化模型
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM,
        plane_resolution=cfg.PLANE_RES, 
        plane_features=cfg.PLANE_FEAT, 
        num_fourier_freqs=cfg.FOURIER_DIM,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM, 
        num_points_curvature=cfg.NUM_POINTS_CURVATURE,
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(DEVICE)

    # 2. 加载权重 (修复 DDP module. 前缀问题)
    try:
        print(f"加载权重: {cfg.MODEL_PATH}")
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 移除 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("✅ 权重对齐并加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return
    model.eval()

    # 3. 三路点云预处理
    pc_data = np.load(cfg.INPUT_PC_PATH)
    u = resample_points(pc_data.get('uniform', np.zeros((0,3))), cfg.NUM_POINTS_UNIFORM)
    c = resample_points(pc_data.get('curvature', np.zeros((0,3))), cfg.NUM_POINTS_CURVATURE)
    i = resample_points(pc_data.get('importance', np.zeros((0,3))), cfg.NUM_POINTS_IMPORTANCE)

    pc_combined = np.concatenate([u, c, i], axis=0)
    pc_tensor = torch.from_numpy(pc_combined).float().unsqueeze(0).to(DEVICE)

    # 4. 执行推理获取 Triplanes
    with torch.no_grad():
        mu, _ = model.encoder(pc_tensor)
        triplanes = model.decoder(mu)
        
        # 如果 Triplanes 返回包含了多余项，只取前三个 (XY, YZ, XZ)
        if isinstance(triplanes, tuple):
            triplanes = triplanes[:3]

        print(f"开始八叉树查询，目标分辨率: {cfg.OCTREE_RESOLUTIONS[-1]}...")
        sdf_grid = octree_refined_sdf(
            model, triplanes, cfg.OCTREE_RESOLUTIONS, DEVICE,
            bbox_size=(cfg.BBOX_MAX - cfg.BBOX_MIN),
            sdf_query_batch_size=cfg.SDF_QUERY_BATCH_SIZE
        )

        # 5. Marching Cubes 提取
        final_res = cfg.OCTREE_RESOLUTIONS[-1]
        sdf_np = sdf_grid.cpu().numpy()
        
        if sdf_np.min() > cfg.MARCHING_CUBES_LEVEL or sdf_np.max() < cfg.MARCHING_CUBES_LEVEL:
            print("⚠️ 警告: 未找到零等值面，请检查模型训练是否收敛。")
            return

        verts, faces, _, _ = measure.marching_cubes(sdf_np, level=cfg.MARCHING_CUBES_LEVEL)
        
        # 坐标映射到 [-1, 1]
        verts = verts * ((cfg.BBOX_MAX - cfg.BBOX_MIN) / final_res) + cfg.BBOX_MIN

        # 6. 导出网格
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        os.makedirs(os.path.dirname(cfg.OUTPUT_MESH_PATH), exist_ok=True)
        mesh.export(cfg.OUTPUT_MESH_PATH)
        print(f"✅ 重建成功! 文件保存在: {cfg.OUTPUT_MESH_PATH}")

if __name__ == '__main__':
    start_t = time.time()
    reconstruct_mesh_with_sdf(Config())
    print(f"总耗时: {time.time() - start_t:.2f}秒")