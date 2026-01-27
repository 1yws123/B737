import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import binary_dilation
from skimage import measure
import time
import torch.nn.functional as F

# 导入你提供的模型类
from vae_model_hash import PointCloudVAE 

# =============================================================================
# 配置参数 (必须与训练时的参数完全一致)
# =============================================================================
class Config:
    MODEL_PATH = r"checkpoint_8(Hash和marchingcubes_npoints1024)/vae_epoch_15000.pth" 
    INPUT_PC_PATH = r"B737_pc/pointcloud/G58_8_pc.npz"
    OUTPUT_MESH_PATH = r"reconstruct/reconstruct_G58_8_8_15000epoch.obj"
    
    # 模型架构参数
    LATENT_DIM = 128
    PLANE_RES = 128
    PLANE_FEAT = 32
    
    
    # 核心：点云规模 (必须与训练时一致)
    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000

    # 八叉树与重建参数
    OCTREE_RESOLUTIONS = [32, 64, 128, 256, 512]
    SDF_QUERY_BATCH_SIZE = 2**16
    MARCHING_CUBES_LEVEL = 0.0
    BLOCK_SIZE = 64
    BBOX_MIN, BBOX_MAX = -1.0, 1.0

# =============================================================================
# 工具函数
# =============================================================================
def resample_points(points, target_num):
    """
    同步 dataset.py 的逻辑：多退少补，确保输入点数固定。
    """
    num_curr = points.shape[0]
    if num_curr == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    
    if num_curr < target_num:
        # 点数不足：随机重复采样
        choice = np.random.choice(num_curr, target_num, replace=True)
        return points[choice]
    else:
        # 点数过多：随机采样
        choice = np.random.choice(num_curr, target_num, replace=False)
        return points[choice]

# =============================================================================
# 高效八叉树 SDF 生成
# =============================================================================
def octree_refined_sdf(model, triplanes, resolutions, device, bbox_size, sdf_query_batch_size, bbox_min, bbox_max):
    coarsest_res = resolutions[0]
    grid_vals = torch.linspace(bbox_min, bbox_max, coarsest_res+1, device=device)
    grid_pts = torch.stack(torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij'), dim=-1)
    
    def query_model(pts):
        # 内部调用 model.py 中的 query_sdf 接口
        return model.query_sdf(triplanes, pts.unsqueeze(0)).squeeze(0).squeeze(-1)

    with torch.no_grad():
        sdf_coarse = query_model(grid_pts.reshape(-1, 3))
    grid_sdf = sdf_coarse.reshape(coarsest_res+1, coarsest_res+1, coarsest_res+1)

    for i in range(len(resolutions)-1):
        cur_res, next_res = resolutions[i], resolutions[i+1]
        step = bbox_size / next_res
        
        # 提取表面附近的区域
        threshold = bbox_size / cur_res
        mask = (grid_sdf.abs() < threshold)
        mask_np = binary_dilation(mask.cpu().numpy(), iterations=1)
        mask = torch.from_numpy(mask_np).to(device)
        
        inds = torch.nonzero(mask, as_tuple=False)
        offset = torch.tensor([[dx,dy,dz] for dx in [0,1] for dy in [0,1] for dz in [0,1]], device=device)
        inds = (inds.unsqueeze(1) * 2 + offset.unsqueeze(0)).reshape(-1, 3)
        inds = torch.unique(inds, dim=0)
        inds = inds[(inds >= 0).all(dim=1) & (inds <= next_res).all(dim=1)]
        
        query_pts = inds.float() * step + bbox_min
        next_logits = torch.full((next_res+1,next_res+1,next_res+1), 10.0, dtype=grid_sdf.dtype, device=device)
        
        for j in range(0, len(inds), sdf_query_batch_size):
            batch_idx = inds[j : j + sdf_query_batch_size]
            batch_pts = query_pts[j : j + sdf_query_batch_size]
            sdf_vals = query_model(batch_pts)
            next_logits[batch_idx[:,0], batch_idx[:,1], batch_idx[:,2]] = sdf_vals
            
        # 保留上一层的采样结果以保持一致性
        factor = next_res // cur_res
        next_logits[::factor, ::factor, ::factor] = grid_sdf
        grid_sdf = next_logits
        
    return grid_sdf

# =============================================================================
# Marching Cubes 表面提取
# =============================================================================
def extract_surface(sdf_grid, cfg: Config):
    final_res = cfg.OCTREE_RESOLUTIONS[-1]
    block_size = cfg.BLOCK_SIZE
    verts_total, faces_total = [], []
    vert_offset = 0
    sdf_np = sdf_grid.cpu().numpy()

    for i in range(0, final_res, block_size):
        for j in range(0, final_res, block_size):
            for k in range(0, final_res, block_size):
                i1, i2 = i, min(i + block_size, final_res)
                j1, j2 = j, min(j + block_size, final_res)
                k1, k2 = k, min(k + block_size, final_res)
                
                block = sdf_np[i1:i2+1, j1:j2+1, k1:k2+1]
                if block.size < 8 or block.min() > cfg.MARCHING_CUBES_LEVEL or block.max() < cfg.MARCHING_CUBES_LEVEL:
                    continue
                
                try:
                    verts, faces, _, _ = measure.marching_cubes(block, level=cfg.MARCHING_CUBES_LEVEL)
                    verts += np.array([i1, j1, k1])
                    verts_total.append(verts)
                    faces_total.append(faces + vert_offset)
                    vert_offset += verts.shape[0]
                except:
                    continue
                
    if not verts_total:
        return None
        
    v = np.concatenate(verts_total, axis=0)
    f = np.concatenate(faces_total, axis=0)
    # 映射回物理空间 BBOX
    v = v * ((cfg.BBOX_MAX - cfg.BBOX_MIN) / final_res) + cfg.BBOX_MIN
    return trimesh.Trimesh(vertices=v, faces=f)

# =============================================================================
# 主流程
# =============================================================================
def reconstruct(cfg: Config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化三路并行模型
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM,
        plane_resolution=cfg.PLANE_RES,
        plane_features=cfg.PLANE_FEAT,
        #num_fourier_freqs=cfg.FOURIER_FREQ,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM,
        num_points_curvature=cfg.NUM_POINTS_CURVATURE,
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(DEVICE)
    
    # 2. 加载权重
    print(f"正在加载模型: {cfg.MODEL_PATH}")
    ckpt = torch.load(cfg.MODEL_PATH, map_location=DEVICE,weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. 处理输入点云 (仿照 dataset.py 加载逻辑)
    print(f"读取点云文件: {cfg.INPUT_PC_PATH}")
    pc_data = np.load(cfg.INPUT_PC_PATH)
    
    # 提取三类点云并进行固定数量重采样
    u_pts = pc_data.get('uniform', pc_data.get('uniform_points', np.zeros((0,3))))
    c_pts = pc_data.get('curvature', np.zeros((0,3))) # 你的报错点：如果 npz 没这个键，这里补 0
    i_pts = pc_data.get('importance', pc_data.get('sharp_points', np.zeros((0,3))))

    # 严格按照 Config 的数量进行重采样
    u_fixed = resample_points(u_pts, cfg.NUM_POINTS_UNIFORM)
    c_fixed = resample_points(c_pts, cfg.NUM_POINTS_CURVATURE)
    i_fixed = resample_points(i_pts, cfg.NUM_POINTS_IMPORTANCE)

    # 按照 [Uniform, Curvature, Importance] 顺序拼接
    pc_combined = np.concatenate([u_fixed, c_fixed, i_fixed], axis=0)
    pc_tensor = torch.from_numpy(pc_combined).float().unsqueeze(0).to(DEVICE)

    # 4. 推理
    with torch.no_grad():
        # Encoder 会根据传入的固定数量参数进行切分处理
        mu, _ = model.encoder(pc_tensor)
        triplanes = model.decoder(mu)

        print("开始八叉树 SDF 采样...")
        sdf_grid = octree_refined_sdf(
            model, triplanes, cfg.OCTREE_RESOLUTIONS, DEVICE,
            bbox_size=(cfg.BBOX_MAX - cfg.BBOX_MIN),
            sdf_query_batch_size=cfg.SDF_QUERY_BATCH_SIZE,
            bbox_min=cfg.BBOX_MIN, bbox_max=cfg.BBOX_MAX
        )

        print("正在提取网格表面...")
        mesh = extract_surface(sdf_grid, cfg)

        if mesh:
            os.makedirs(os.path.dirname(cfg.OUTPUT_MESH_PATH), exist_ok=True)
            mesh.export(cfg.OUTPUT_MESH_PATH)
            print(f"✅ 重建完成: {cfg.OUTPUT_MESH_PATH}")
        else:
            print("❌ 重建失败：未提取到表面")

if __name__ == '__main__':
    start = time.time()
    reconstruct(Config())
    print(f"总耗时: {time.time() - start:.2f} 秒")