import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm
import pathlib
import open3d as o3d

def fps_downsample(points, num_required):
    """
    使用 Open3D 的最远点采样 (FPS)。
    """
    if points.shape[0] == 0:
        return np.zeros((num_required, 3), dtype=np.float32)
        
    if points.shape[0] <= num_required:
        # 如果点数不够，直接返回并随机补齐
        indices = np.random.choice(points.shape[0], num_required, replace=True)
        return points[indices]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.farthest_point_down_sample(num_required)
    return np.asarray(pcd_down.points)

def process_model(model_path, output_root_dir, num_points=4000, 
                  subdivision_iterations=2, sharp_edge_angle_threshold_deg=15.0):
    try:
        # 1. 加载原始模型
        mesh = trimesh.load(model_path, force='mesh')
        trimesh.repair.fix_normals(mesh)

        # 2. 全局归一化 (Normalization)
        # 注意：所有操作都在归一化后的坐标系下进行，确保三个点云对齐
        v = mesh.vertices
        shifts = (v.max(axis=0) + v.min(axis=0)) / 2
        v = v - shifts
        scale = (1 / np.abs(v).max()) * 0.9
        v = v * scale
        mesh.vertices = v

        # =========================================================
        # A. Curvature 点云 (基于曲率: 减面 -> 计算 -> 前60% -> FPS)
        # =========================================================
        # 创建一个副本用于减面和曲率计算，不影响原始 mesh
        mesh_for_curvature = mesh.copy()

        # 减面 (为了计算效率和特征提取准确性)
        if len(mesh_for_curvature.faces) > 100000:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_for_curvature.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_for_curvature.faces)
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
            mesh_for_curvature = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                                 faces=np.asarray(o3d_mesh.triangles))
            trimesh.repair.fix_normals(mesh_for_curvature)

        # 计算曲率
        radius = 0.02
        curvatures = trimesh.curvature.discrete_mean_curvature_measure(mesh_for_curvature, mesh_for_curvature.vertices, radius)
        
        # 过滤凹面/连接处，只保留凸面特征 (max(0))
        curvature_score = np.maximum(curvatures, 0)
        
        # 排序并截取前 60%
        sorted_indices = np.argsort(curvature_score)[::-1]
        split_idx = int(len(sorted_indices) * 0.60)
        pool_curvature = mesh_for_curvature.vertices[sorted_indices[:split_idx]]
        
        print(f"  - [Curvature] 从减面后网格的前60%高曲率区域采样 {num_points} 点...")
        curvature_points = fps_downsample(pool_curvature, num_points)

        # =========================================================
        # B. Uniform 点云 (原始 Mesh -> 细分 -> FPS)
        # =========================================================
        # 使用原始(归一化后) Mesh
        subdivided_mesh = mesh.copy()
        for _ in range(subdivision_iterations):
            subdivided_mesh = subdivided_mesh.subdivide()
        uniform_pool = subdivided_mesh.vertices
        
        print(f"  - [Uniform] 从细分后的原始网格采样 {num_points} 点...")
        uniform_points = fps_downsample(uniform_pool, num_points)

        # =========================================================
        # C. Importance 点云 (原始 Mesh -> 锐利边 -> FPS)
        # =========================================================
        # 使用原始(归一化后) Mesh
        angle_rad = np.deg2rad(sharp_edge_angle_threshold_deg)
        # 计算面邻接角度
        edge_angles = mesh.face_adjacency_angles
        # 筛选锐利边
        sharp_edge_indices = mesh.face_adjacency_edges[edge_angles > angle_rad]
        
        if len(sharp_edge_indices) > 0:
            # 获取锐利边的顶点坐标 shape: (num_edges, 2, 3)
            sharp_lines = mesh.vertices[sharp_edge_indices] 
            
            # 在线段上随机采样生成候选池
            num_pool_samples = 100000
            line_indices = np.random.randint(0, len(sharp_lines), size=num_pool_samples)
            t = np.random.random(size=(num_pool_samples, 1)).astype(np.float32)
            # 线性插值
            importance_pool = (1 - t) * sharp_lines[line_indices, 0] + t * sharp_lines[line_indices, 1]
        else:
            print(f"  - [警告] {os.path.basename(model_path)} 未找到锐利边，使用 Uniform 池替代。")
            importance_pool = uniform_pool

        print(f"  - [Importance] 从锐利边采样 {num_points} 点...")
        importance_points = fps_downsample(importance_pool, num_points)

        # 4. 保存结果
        p = pathlib.Path(model_path)
        model_id = p.stem
        category_name = p.parent.parent.name if p.parent.parent.name else "default"
            
        final_output_dir = os.path.join(output_root_dir, category_name, 'pointcloud')
        os.makedirs(final_output_dir, exist_ok=True)
        
        output_path = os.path.join(final_output_dir, f"{model_id}_pc_3types.npz")
        np.savez(
            output_path,
            uniform=uniform_points.astype(np.float32),
            curvature=curvature_points.astype(np.float32),
            importance=importance_points.astype(np.float32)
        )
        print(f"  - 保存成功: {output_path}")
        
    except Exception as e:
        print(f"处理文件 {model_path} 时发生错误: {e}")

if __name__ == '__main__':
    ORIGINAL_PC_ROOT = r"F:\Code\mmm\reconstruct"
    NEW_PC_ROOT_DIR = r"F:\Code\mmm"

    # 根据你的路径调整
    glob_pattern = os.path.join(ORIGINAL_PC_ROOT, 'G58_32_cleaned.obj')
    model_files = glob.glob(glob_pattern)
    
    if not model_files:
        print(f"错误: 未找到文件 '{glob_pattern}'")
    else:
        for model_path in tqdm(model_files, desc="Processing"):
            process_model(
                model_path, 
                NEW_PC_ROOT_DIR, 
                num_points=4000
            )