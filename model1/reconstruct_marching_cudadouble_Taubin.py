import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import binary_dilation
from skimage import measure
import time

from model import PointCloudVAE


def keep_largest_component(mesh):
    """
    核心：只保留顶点数量最多的那个组件，剔除所有漂浮的乱线和细碎面片。
    """
    # split 会将互不相连的几何体拆分开
    # only_watertight=False 确保非封闭的网格也能被拆分
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # 按顶点数从大到小排序，飞机主体通常是第0个
        components = sorted(components, key=lambda x: len(x.vertices), reverse=True)
        print(f"--- 检测到 {len(components)} 个独立组件，已自动删除 {len(components)-1} 个碎片 ---")
        return components[0]
    return mesh

def post_process_mesh(vertices, faces, iterations=20, lamb=0.5, mu=-0.53):
    """
    对生成的网格应用 Taubin 平滑处理
    参数:
        vertices: 顶点数组 (N, 3)
        faces: 面片数组 (M, 3)
        iterations: 迭代次数。次数越多越平滑，但也可能导致精细特征丢失。
        lamb: 正向平滑因子 (通常取 0.5)
        mu: 反向膨胀因子 (通常取 -0.53，要求 mu < -lamb 才能防止收缩)
    返回:
        smoothed_mesh: 处理后的 trimesh 对象
    """
    # 1. 创建网格并利用内置的 process=True 进行基础清理（已包含去重、合并等）
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    if len(mesh.vertices) == 0:
        print("错误: 提取的初始网格为空")
        return mesh

    # 2. 移除无效面（解决报错：不再直接调用 remove_duplicate_faces）
    # 在最新版 trimesh 中，可以使用以下替代方案：
    mesh.update_faces(mesh.nondegenerate_faces()) # 移除面积为0的面
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    # 3. 【关键步骤】剔除乱线：只保留飞机主体
    print("正在进行连通域分析以切断乱线...")
    mesh = keep_largest_component(mesh)

    # 4. Taubin 光顺处理（解决毛刺和台阶感）
    print(f"正在应用 Taubin 平滑 (迭代: {iterations})...")
    try:
        import inspect
        sig = inspect.signature(trimesh.smoothing.filter_taubin)
        # 自动适配参数名：旧版用 mu, 新版可能用 nu
        params = {'iterations': iterations, 'lamb': lamb}
        if 'nu' in sig.parameters:
            params['nu'] = mu
        else:
            params['mu'] = mu
            
        mesh = trimesh.smoothing.filter_taubin(mesh, **params)
    except Exception as e:
        print(f"Taubin 平滑失败 ({e}), 回退至基础 Laplacian 平滑")
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)

    return mesh
# =============================================================================
class Config:
    # 确保这里的路径指向您最新的权重（包含 Curvature 分支的训练结果）
    MODEL_PATH = r"checkpoint_10(Fourier和marchingcubes,fps1024,sdf=40w)/vae_epoch_10000.pth" 
    INPUT_PC_PATH = r"B737_pc/pointcloud/G58_8_pc.npz"
    OUTPUT_MESH_PATH = r"reconstruct/reconstruct_G58_8_10_10000epoch_2.obj"

    # 模型结构参数
    LATENT_DIM = 128
    PLANE_RES =128
    PLANE_FEAT = 32
    FOURIER_DIM = 8

    # 【重要】必须与 train.py 中的设置完全一致
    # 如果训练时用了 4000, 4000, 4000，这里必须也是 4000
    NUM_POINTS_UNIFORM = 4000
    NUM_POINTS_CURVATURE = 4000
    NUM_POINTS_IMPORTANCE = 4000

    # Marching Cubes & Octree 参数
    OCTREE_RESOLUTIONS = [32, 64, 128, 256,512] # 测试时可以先开小一点，比如到 256
    SDF_QUERY_BATCH_SIZE = 65536
    MARCHING_CUBES_LEVEL = 0.00005
    BLOCK_SIZE = 64
    BBOX_MIN, BBOX_MAX = -1.0, 1.0

# =============================================================================
# 辅助函数：重采样 (多退少补)
# =============================================================================
def resample_points(points, target_num):
    """
    确保点云数量严格等于 target_num。
    如果点数不足则随机重复，点数过多则随机截取。
    """
    num_curr = points.shape[0]
    if num_curr == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    
    if num_curr < target_num:
        # 点数不足：随机重复采样补齐
        choice = np.random.choice(num_curr, target_num, replace=True)
        return points[choice]
    elif num_curr > target_num:
        # 点数过多：随机截取
        choice = np.random.choice(num_curr, target_num, replace=False)
        return points[choice]
    else:
        return points

# =============================================================================
# 高效八叉树 refinement SDF生成 (保持不变)
# =============================================================================
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
        # 找near-surface mask
        threshold = bbox_size / cur_res
        mask = (grid_sdf.abs() < threshold)
        mask_np = mask.cpu().numpy()
        mask_np = binary_dilation(mask_np, iterations=1)
        mask = torch.from_numpy(mask_np).to(device)
        # 全批量生成所有8子格索引
        inds = torch.nonzero(mask, as_tuple=False)
        offset = torch.tensor([[dx,dy,dz] for dx in [0,1] for dy in [0,1] for dz in [0,1]], device=device)
        inds = inds.unsqueeze(1)*2 + offset.unsqueeze(0)
        inds = inds.reshape(-1,3)
        valid = (inds>=0) & (inds<=next_res)
        inds = inds[valid.all(dim=1)]
        inds = torch.unique(inds, dim=0)
        # SDF批量推理
        query_pts = inds.float()*step + grid_min
        next_logits = torch.full((next_res+1,next_res+1,next_res+1), 10000.0, dtype=grid_sdf.dtype, device=device)
        for j in range(0,len(inds),sdf_query_batch_size):
            pts_batch = query_pts[j:j+sdf_query_batch_size].unsqueeze(0)
            sdf_vals = model.query_sdf(triplanes, pts_batch).squeeze(0).squeeze(-1)
            idx = inds[j:j+sdf_query_batch_size]
            next_logits[idx[:,0], idx[:,1], idx[:,2]] = sdf_vals
        # 低分辨采样点逐点保留
        factor = next_res // cur_res
        next_logits[::factor,::factor,::factor] = grid_sdf
        grid_sdf = next_logits
    return grid_sdf

# =============================================================================
# 分块 marching cubes mesh 提取 (保持不变)
# =============================================================================
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

# =============================================================================
# 主重建流程（带SDF shift/scale归一化）
# =============================================================================
def reconstruct_mesh_with_sdf(cfg: Config):
    DEVICE = torch.device( "cpu")
    print(f"使用的设备: {DEVICE}")

    # 1. 初始化模型：传入三个点数参数
    print("正在初始化 VAE 模型...")
    model = PointCloudVAE(
        latent_dim=cfg.LATENT_DIM,
        plane_resolution=cfg.PLANE_RES, 
        plane_features=cfg.PLANE_FEAT, 
        num_fourier_freqs=cfg.FOURIER_DIM,
        num_points_uniform=cfg.NUM_POINTS_UNIFORM, 
        num_points_curvature=cfg.NUM_POINTS_CURVATURE, # 新增
        num_points_importance=cfg.NUM_POINTS_IMPORTANCE
    ).to(DEVICE)

    # 2. 加载权重
    try:
        print(f"加载权重: {cfg.MODEL_PATH}")
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE,weights_only=False)
        # 兼容只保存了 state_dict 或者是完整 checkpoint 的情况
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' (长度为7)
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        # ------------------------------------------

    except Exception as e:
        print(f"[错误] 加载权重失败: {e}")
        return
        
    model.eval()

    # 3. 加载并预处理点云
    print(f"加载点云: {cfg.INPUT_PC_PATH}")
    pc_data = np.load(cfg.INPUT_PC_PATH)
    
    # 获取三种类型的点 (兼容旧文件，如果没有则补0)
    u_pts_raw = pc_data['uniform'] if 'uniform' in pc_data else np.zeros((0, 3))
    c_pts_raw = pc_data['curvature'] if 'curvature' in pc_data else np.zeros((0, 3))
    i_pts_raw = pc_data['importance'] if 'importance' in pc_data else np.zeros((0, 3))

    # 【核心步骤】重采样到 Config 中定义的固定数量
    # 必须与 model.encoder 中的切分逻辑严丝合缝
    u_pts = resample_points(u_pts_raw, cfg.NUM_POINTS_UNIFORM)
    c_pts = resample_points(c_pts_raw, cfg.NUM_POINTS_CURVATURE)
    i_pts = resample_points(i_pts_raw, cfg.NUM_POINTS_IMPORTANCE)

    print(f"点云统计 (Resampled):")
    print(f"  - Uniform: {u_pts.shape}")
    print(f"  - Curvature: {c_pts.shape}")
    print(f"  - Importance: {i_pts.shape}")

    # 按顺序拼接：Uniform -> Curvature -> Importance
    point_cloud_full = np.concatenate([u_pts, c_pts, i_pts], axis=0)
    
    # 转为 Tensor: [1, N, 3]
    point_cloud_tensor = torch.from_numpy(point_cloud_full).float().unsqueeze(0).to(DEVICE)

    # 4. 推理
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

        final_mesh = post_process_mesh(vertices, faces, iterations=20)
    # ---------------------

    # 导出
        print(f"导出已平滑的 mesh obj: {cfg.OUTPUT_MESH_PATH}")
        os.makedirs(os.path.dirname(cfg.OUTPUT_MESH_PATH), exist_ok=True)
        final_mesh.export(cfg.OUTPUT_MESH_PATH)
        print(f"✅ 网格成功保存并完成光顺处理")



if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    reconstruct_mesh_with_sdf(config)
    print(f"\n总耗时: {(time.time()-start_time):.2f}秒")