import torch
import numpy as np
import os
import trimesh
import time
from flexicubes import FlexiCubes

# 导入你最新的模型类 (确保路径正确)
from model_flexicube import PointCloudVAE

# =============================================================================
# 配置参数
# =============================================================================
class Config:
    # 路径配置
    MODEL_PATH = r"/home/yuwenshi/B737/checkpoint_4(Fourier和flexicubes)/vae_epoch_8000.pth"
    INPUT_PC_PATH = r"B737_pc/pointcloud/G58_8_pc.npz"
    OUTPUT_MESH_PATH = r"reconstruct/reconstruct_G58_8_4_8000_1024.obj"
    
    # 模型架构参数 (必须与训练时严格一致)
    LATENT_DIM = 128
    PLANE_RES = 128
    PLANE_FEAT = 32
    FOURIER_DIM = 8
    
    # 核心：三路点云数量配置
    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000
    
    # FlexiCubes 重建分辨率
    GRID_RES = 1024 
    QUERY_BATCH_SIZE = 100000 

# =============================================================================
# 辅助工具
# =============================================================================
def resample_points(points, target_num):
    """同步训练时的重采样逻辑：多退少补"""
    num_curr = points.shape[0]
    if num_curr == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    
    if num_curr < target_num:
        choice = np.random.choice(num_curr, target_num, replace=True)
        return points[choice]
    else:
        choice = np.random.choice(num_curr, target_num, replace=False)
        return points[choice]

def construct_grid_topology(res, device):
    """构建 FlexiCubes 所需的网格拓扑"""
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

# =============================================================================
# 主重建流程
# =============================================================================
def reconstruct_mesh_flexicubes(cfg: Config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {DEVICE}")

    # 1. 加载模型 (传入三个点云数量参数)
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM, 
        plane_resolution=cfg.PLANE_RES, 
        plane_features=cfg.PLANE_FEAT, 
        num_fourier_freqs=cfg.FOURIER_DIM,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM,
        num_points_curvature=cfg.NUM_POINTS_CURVATURE,
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(DEVICE)
    
    try:
        print(f"加载权重: {cfg.MODEL_PATH}")
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"错误: 无法加载权重 - {e}")
        return
        
    model.eval()

    # 2. 初始化 FlexiCubes
    fc = FlexiCubes(device=DEVICE)
    voxel_verts, voxel_cubes = construct_grid_topology(cfg.GRID_RES, DEVICE)
    voxel_verts = voxel_verts.float()
    voxel_cubes = voxel_cubes.long()
    
    # 3. 处理三路点云数据
    print(f"读取并处理三路点云: {cfg.INPUT_PC_PATH}")
    pc_data = np.load(cfg.INPUT_PC_PATH)
    
    # 分别提取并采样，确保数量对齐
    u_pts = pc_data.get('uniform', pc_data.get('uniform_points', np.zeros((0,3))))
    c_pts = pc_data.get('curvature', np.zeros((0,3)))
    i_pts = pc_data.get('importance', pc_data.get('sharp_points', np.zeros((0,3))))

    u_fixed = resample_points(u_pts, cfg.NUM_POINTS_UNIFORM)
    c_fixed = resample_points(c_pts, cfg.NUM_POINTS_CURVATURE)
    i_fixed = resample_points(i_pts, cfg.NUM_POINTS_IMPORTANCE)

    # 按照训练时的顺序 [Uniform, Curvature, Importance] 拼接
    pc_combined = np.concatenate([u_fixed, c_fixed, i_fixed], axis=0)
    pc_tensor = torch.from_numpy(pc_combined).float().unsqueeze(0).to(DEVICE)

    # 4. 推理
    with torch.no_grad():
        # Encoder -> Triplanes & FlexiCubes Weights
        # 这里的 model 返回值需与你的 model_flexicube.py 保持一致
        outputs = model(pc_tensor)
        triplanes = outputs[0]
        flex_weights = outputs[-1] # 假设 w 是最后一个返回值
        
        # 解析权重 (Beta, Alpha, Gamma)
        w = flex_weights[0] 
        n_cubes = voxel_cubes.shape[0]
        beta = w[:12].unsqueeze(0).expand(n_cubes, -1)
        alpha = w[12:20].unsqueeze(0).expand(n_cubes, -1)
        gamma = w[20].unsqueeze(0).expand(n_cubes)
        
        # 5. 分 Batch 查询 Grid SDF
        print(f"正在查询 Grid SDF (Total points: {voxel_verts.shape[0]})...")
        num_verts = voxel_verts.shape[0]
        grid_sdf_list = []
        for i in range(0, num_verts, cfg.QUERY_BATCH_SIZE):
            batch_verts = voxel_verts[i : i + cfg.QUERY_BATCH_SIZE].unsqueeze(0)
            pred_sdf = model.query_sdf(triplanes, batch_verts).squeeze(-1)
            grid_sdf_list.append(pred_sdf)
            
        grid_sdf = torch.cat(grid_sdf_list, dim=1).squeeze(0)
        
        # 6. FlexiCubes 提取网格
        print("执行 FlexiCubes 提取...")
        mesh_verts, mesh_faces, _ = fc(
            x_nx3=voxel_verts,
            s_n=grid_sdf,
            cube_fx8=voxel_cubes,
            res=cfg.GRID_RES,
            beta_fx12=beta,
            alpha_fx8=alpha,
            gamma_f=gamma,
            training=False
        )
        
        if mesh_verts.shape[0] == 0:
            print("❌ 重建失败：Mesh 顶点数为 0。请检查模型输出的 SDF 范围。")
            return

        # 7. 保存结果
        vertices = mesh_verts.cpu().numpy()
        faces = mesh_faces.cpu().numpy()
        
        os.makedirs(os.path.dirname(cfg.OUTPUT_MESH_PATH), exist_ok=True)
        # FlexiCubes 生成的通常是四边形网格，trimesh 会自动处理或你可以保持 process=False
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(cfg.OUTPUT_MESH_PATH)
        
        print(f"✅ 重建成功! 结果保存至: {cfg.OUTPUT_MESH_PATH}")
        print(f"统计: {vertices.shape[0]} 顶点, {faces.shape[0]} 面片")

if __name__ == '__main__':
    start_t = time.time()
    reconstruct_mesh_flexicubes(Config())
    print(f"\n任务耗时: {(time.time() - start_t):.2f}s")