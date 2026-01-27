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

        # 2. 全局归一化
        v = mesh.vertices
        shifts = (v.max(axis=0) + v.min(axis=0)) / 2
        v = v - shifts
        scale = (1 / np.max(np.abs(v))) * 0.9 # 修正了 np.abs(v).max() 的写法
        v = v * scale
        mesh.vertices = v

        # --- A. Curvature 点云 ---
        mesh_for_curvature = mesh.copy()
        if len(mesh_for_curvature.faces) > 100000:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_for_curvature.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_for_curvature.faces)
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
            mesh_for_curvature = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                                 faces=np.asarray(o3d_mesh.triangles))
            trimesh.repair.fix_normals(mesh_for_curvature)

        radius = 0.02
        curvatures = trimesh.curvature.discrete_mean_curvature_measure(mesh_for_curvature, mesh_for_curvature.vertices, radius)
        curvature_score = np.maximum(curvatures, 0)
        sorted_indices = np.argsort(curvature_score)[::-1]
        split_idx = int(len(sorted_indices) * 0.60)
        pool_curvature = mesh_for_curvature.vertices[sorted_indices[:split_idx]]
        curvature_points = fps_downsample(pool_curvature, num_points)

        # --- B. Uniform 点云 ---
        subdivided_mesh = mesh.copy()
        for _ in range(subdivision_iterations):
            subdivided_mesh = subdivided_mesh.subdivide()
        uniform_pool = subdivided_mesh.vertices
        uniform_points = fps_downsample(uniform_pool, num_points)

        # --- C. Importance 点云 ---
        angle_rad = np.deg2rad(sharp_edge_angle_threshold_deg)
        edge_angles = mesh.face_adjacency_angles
        sharp_edge_indices = mesh.face_adjacency_edges[edge_angles > angle_rad]
        
        if len(sharp_edge_indices) > 0:
            sharp_lines = mesh.vertices[sharp_edge_indices] 
            num_pool_samples = 100000
            line_indices = np.random.randint(0, len(sharp_lines), size=num_pool_samples)
            t = np.random.random(size=(num_pool_samples, 1)).astype(np.float32)
            importance_pool = (1 - t) * sharp_lines[line_indices, 0] + t * sharp_lines[line_indices, 1]
        else:
            importance_pool = uniform_pool
        importance_points = fps_downsample(importance_pool, num_points)

        # 4. 保存结果
        model_id = pathlib.Path(model_path).stem
        # 直接保存在输出根目录下的 pointcloud 文件夹中
        final_output_dir = os.path.join(output_root_dir, 'pointcloud')
        os.makedirs(final_output_dir, exist_ok=True)
        
        output_path = os.path.join(final_output_dir, f"{model_id}_pc.npz")
        np.savez(
            output_path,
            uniform=uniform_points.astype(np.float32),
            curvature=curvature_points.astype(np.float32),
            importance=importance_points.astype(np.float32)
        )
        
    except Exception as e:
        print(f"\n处理文件 {os.path.basename(model_path)} 时发生错误: {e}")

if __name__ == '__main__':
    # 修改后的输入输出路径
    INPUT_MESH_DIR = r"B737_500"
    OUTPUT_PC_ROOT = r"B737_pc"

    # 获取目录下所有 obj 文件
    model_files = glob.glob(os.path.join(INPUT_MESH_DIR, "*.obj"))
    
    if not model_files:
        print(f"错误: 在 '{INPUT_MESH_DIR}' 未找到任何 .obj 文件")
    else:
        print(f"找到 {len(model_files)} 个文件，开始处理...")
        for model_path in tqdm(model_files, desc="Batch Processing"):
            process_model(
                model_path, 
                OUTPUT_PC_ROOT, 
                num_points=4000
            )
        print(f"\n所有任务完成！结果保存在: {os.path.join(OUTPUT_PC_ROOT, 'pointcloud')}")