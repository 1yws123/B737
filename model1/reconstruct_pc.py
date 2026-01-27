import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import binary_dilation
from skimage import measure
import time

from model import PointCloudVAE

# =============================================================================
# 配置参数
# =============================================================================
class Config:
    MODEL_PATH = r"/home/yuwenshi/B737/checkpoint_6(Fourier和marchingcubes)/vae_epoch_10000.pth" 
    INPUT_PC_PATH = r"B737_pc/pointcloud/G58_8_pc.npz"
    OUTPUT_PC_PATH = r"reconstruct/reconstruct_G58_8_6_10000epoch.ply"  # 支持 .npz 或 .ply

    LATENT_DIM = 128
    PLANE_RES =128
    PLANE_FEAT = 32
    FOURIER_DIM = 8

    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000

    OCTREE_RESOLUTIONS = [32, 64, 128, 256, 512]
    SDF_QUERY_BATCH_SIZE = 65536
    MARCHING_CUBES_LEVEL = 0.0
    BLOCK_SIZE = 64
    BBOX_MIN, BBOX_MAX = -1.0, 1.0

    TARGET_POINT_NUM = 12000

def resample_points(points, target_num):
    num_curr = points.shape[0]
    if num_curr == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    if num_curr < target_num:
        choice = np.random.choice(num_curr, target_num, replace=True)
        return points[choice]
    elif num_curr > target_num:
        choice = np.random.choice(num_curr, target_num, replace=False)
        return points[choice]
    else:
        return points

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
        threshold = bbox_size / cur_res
        mask = (grid_sdf.abs() < threshold)
        mask_np = mask.cpu().numpy()
        mask_np = binary_dilation(mask_np, iterations=1)
        mask = torch.from_numpy(mask_np).to(device)
        inds = torch.nonzero(mask, as_tuple=False)
        offset = torch.tensor([[dx,dy,dz] for dx in [0,1] for dy in [0,1] for dz in [0,1]], device=device)
        inds = inds.unsqueeze(1)*2 + offset.unsqueeze(0)
        inds = inds.reshape(-1,3)
        valid = (inds>=0) & (inds<=next_res)
        inds = inds[valid.all(dim=1)]
        inds = torch.unique(inds, dim=0)
        query_pts = inds.float()*step + grid_min
        next_logits = torch.full((next_res+1,next_res+1,next_res+1), 10000.0, dtype=grid_sdf.dtype, device=device)
        for j in range(0,len(inds),sdf_query_batch_size):
            pts_batch = query_pts[j:j+sdf_query_batch_size].unsqueeze(0)
            sdf_vals = model.query_sdf(triplanes, pts_batch).squeeze(0).squeeze(-1)
            idx = inds[j:j+sdf_query_batch_size]
            next_logits[idx[:,0], idx[:,1], idx[:,2]] = sdf_vals
        factor = next_res // cur_res
        next_logits[::factor,::factor,::factor] = grid_sdf
        grid_sdf = next_logits
    return grid_sdf

def extract_surface_marching_cubes(sdf_grid, cfg: Config):
    final_res = cfg.OCTREE_RESOLUTIONS[-1]
    block_size = cfg.BLOCK_SIZE
    verts_total, faces_total = [], []
    vert_offset = 0

    sdf_np = sdf_grid.cpu().numpy()
    for i in range(0, final_res, block_size):
        for j in range(0, final_res, block_size):
            for k in range(0, final_res, block_size):
                i1, i2 = i, min(i+block_size, final_res)
                j1, j2 = j, min(j+block_size, final_res)
                k1, k2 = k, min(k+block_size, final_res)
                block = sdf_np[i1:i2+1, j1:j2+1, k1:k2+1]
                if block.min() > cfg.MARCHING_CUBES_LEVEL or block.max() < cfg.MARCHING_CUBES_LEVEL:
                    continue
                verts, faces, _, _ = measure.marching_cubes(block, level=cfg.MARCHING_CUBES_LEVEL)
                if verts.shape[0] == 0: continue
                verts += np.array([i1, j1, k1])
                verts_total.append(verts)
                faces_total.append(faces + vert_offset)
                vert_offset += verts.shape[0]
    if not verts_total:
        print("未提取到 mesh。")
        return np.zeros((0,3)), np.zeros((0,3), dtype=int)
    verts_total = np.concatenate(verts_total, axis=0)
    faces_total = np.concatenate(faces_total, axis=0)
    scale = (cfg.BBOX_MAX - cfg.BBOX_MIN) / final_res
    offset = cfg.BBOX_MIN
    verts_total = verts_total * scale + offset
    return verts_total, faces_total

def reconstruct_mesh_and_sample_pointcloud(cfg: Config):
    DEVICE = torch.device( "cpu")
    print(f"使用的设备: {DEVICE}")

    print("正在初始化 VAE 模型...")
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM,
        plane_resolution=cfg.PLANE_RES, 
        plane_features=cfg.PLANE_FEAT, 
        num_fourier_freqs=cfg.FOURIER_DIM,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM, 
        num_points_curvature=cfg.NUM_POINTS_CURVATURE,
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(DEVICE)

    # load weights
    try:
        print(f"加载权重: {cfg.MODEL_PATH}")
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE,weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"[错误] 加载权重失败: {e}")
        return
        
    model.eval()

    # load input point cloud (with 3 types for new model)
    print(f"加载点云: {cfg.INPUT_PC_PATH}")
    pc_data = np.load(cfg.INPUT_PC_PATH)
    u_pts_raw = pc_data['uniform'] if 'uniform' in pc_data else np.zeros((0, 3))
    c_pts_raw = pc_data['curvature'] if 'curvature' in pc_data else np.zeros((0, 3))
    i_pts_raw = pc_data['importance'] if 'importance' in pc_data else np.zeros((0, 3))
    u_pts = resample_points(u_pts_raw, cfg.NUM_POINTS_UNIFORM)
    c_pts = resample_points(c_pts_raw, cfg.NUM_POINTS_CURVATURE)
    i_pts = resample_points(i_pts_raw, cfg.NUM_POINTS_IMPORTANCE)
    print(f"点云统计 (Resampled):")
    print(f"  - Uniform: {u_pts.shape}")
    print(f"  - Curvature: {c_pts.shape}")
    print(f"  - Importance: {i_pts.shape}")

    point_cloud_full = np.concatenate([u_pts, c_pts, i_pts], axis=0)
    point_cloud_tensor = torch.from_numpy(point_cloud_full).float().unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        mu, _ = model.encoder(point_cloud_tensor)
        decoder_output = model.decoder(mu)
        if isinstance(decoder_output, tuple) and len(decoder_output) == 4:
            plane_xy, plane_yz, plane_xz, _ = decoder_output
        else:
            plane_xy, plane_yz, plane_xz = decoder_output
        triplanes = (plane_xy, plane_yz, plane_xz)

        print("生成SDF八叉树体素...")
        sdf_grid = octree_refined_sdf(
            model, triplanes, cfg.OCTREE_RESOLUTIONS, DEVICE,
            bbox_size=(cfg.BBOX_MAX-cfg.BBOX_MIN),
            sdf_query_batch_size=cfg.SDF_QUERY_BATCH_SIZE
        )
        print("SDF体素生成完成，开始提取表面...")

        vertices, faces = extract_surface_marching_cubes(sdf_grid, cfg)
        if vertices.shape[0] == 0:
            print("重建失败，没有生成任何网格。")
            return

        print("开始 mesh 表面采样点云...")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        pc_pts, _ = trimesh.sample.sample_surface(mesh, cfg.TARGET_POINT_NUM)
        # 如果采样返回点数不够，兜底补齐
        if pc_pts.shape[0] < cfg.TARGET_POINT_NUM:
            print("表面采样点数不足，补齐...")
            pc_pts = np.concatenate(
                [pc_pts, pc_pts[np.random.choice(pc_pts.shape[0], cfg.TARGET_POINT_NUM - pc_pts.shape[0], replace=True)]],
                axis=0
            )

        print(f"保存采样点云 ({pc_pts.shape[0]} points): {cfg.OUTPUT_PC_PATH}")
        os.makedirs(os.path.dirname(cfg.OUTPUT_PC_PATH), exist_ok=True)
        if cfg.OUTPUT_PC_PATH.endswith('.npz'):
            np.savez(cfg.OUTPUT_PC_PATH, point_cloud=pc_pts.astype(np.float32))
        elif cfg.OUTPUT_PC_PATH.endswith('.ply'):
            # 用trimesh导出纯点云
            pc_mesh = trimesh.points.PointCloud(pc_pts)
            pc_mesh.export(cfg.OUTPUT_PC_PATH)
        else:
            np.savetxt(cfg.OUTPUT_PC_PATH, pc_pts, delimiter=',', header='x,y,z', comments='')

        print(f"✅ 点云已保存: {cfg.OUTPUT_PC_PATH} (点数: {pc_pts.shape[0]})")

if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    reconstruct_mesh_and_sample_pointcloud(config)
    print(f"\n总耗时: {(time.time()-start_time):.2f}秒")