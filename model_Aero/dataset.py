import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import re

class SDFDataset(Dataset):
    def __init__(self, pc_root_dir, aero_root_dir, sdf_dir, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, 
                 num_points_importance=4000,
                 num_points_sdf=16384, 
                 surface_ratio=0.8,
                 surface_threshold=0.02):
        """
        Args:
            pc_root_dir: 点云文件所在目录 (/home/yuwenshi/B737/G58_mesh_1299)
            aero_root_dir: 气动数据根目录 (/home/yuwenshi/B737/G58_aero_1299/G58_aero_1299)
            sdf_dir: SDF .npz 文件所在目录
        """
        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature
        self.num_points_importance = num_points_importance
        self.num_points_pc = num_points_uniform + num_points_curvature + num_points_importance
        
        self.num_points_sdf = num_points_sdf
        self.surface_ratio = surface_ratio
        self.surface_threshold = surface_threshold
        
        self.pc_root_dir = pc_root_dir
        self.aero_root_dir = aero_root_dir
        self.sdf_dir = sdf_dir

        # 执行自动匹配逻辑
        self.file_pairs = self._make_dataset()
        
        if not self.file_pairs:
            raise RuntimeError(f"无法匹配数据！请检查路径：\nPC: {pc_root_dir}\nAero: {aero_root_dir}\nSDF: {sdf_dir}")
        
        print(f"✅ Dataset 初始化成功!")
        print(f"   找到匹配样本总数: {len(self.file_pairs)}")

    def _make_dataset(self):
        """
        核心匹配逻辑：以 Mesh 文件夹中的 ID 为基准，寻找对应的 SDF 和 Polar 文件
        """
        # 获取所有点云文件路径
        pc_files = glob.glob(os.path.join(self.pc_root_dir, 'G58_*_pc.npz'))
        file_pairs = []

        for pc_path in pc_files:
            # 1. 提取 ID (例如 G58_1)
            file_name = os.path.basename(pc_path)
            # 使用正则匹配 G58_ 后面跟着的数字
            match = re.search(r'(G58_\d+)', file_name)
            if not match:
                continue
            file_id = match.group(1)

            # 2. 构建对应的 SDF 路径
            sdf_path = os.path.join(self.sdf_dir, f"{file_id}.npz")

            # 3. 构建气动 Polar 路径
            # 根据你的路径结构：aero_root/G58_1/G58_1_VSPGeom.polar
            polar_path = os.path.join(self.aero_root_dir, file_id, f"{file_id}_VSPGeom.polar")

            # 4. 只有当三者同时存在时，才加入训练列表
            if os.path.exists(sdf_path) and os.path.exists(polar_path):
                file_pairs.append({
                    'pc_path': pc_path,
                    'sdf_path': sdf_path,
                    'polar_path': polar_path,
                    'file_id': file_id
                })
        
        # 按 ID 数字顺序排序（可选，方便调试）
        file_pairs.sort(key=lambda x: int(x['file_id'].split('_')[1]))
        return file_pairs

    def _parse_polar(self, polar_path):
        """
        专门解析 VSPGeom.polar 文件
        根据 G58_1_VSPGeom.polar：
        CL 是第 5 列 (index 4)
        CDtot_t 是第 10 列 (index 9)
        """
        try:
            with open(polar_path, 'r') as f:
                lines = f.readlines()
                # 寻找包含数据的那一行（通常是最后一行）
                data_line = lines[-1].strip()
                if not data_line: # 防止末尾空行
                    data_line = lines[-2].strip()
                
                parts = data_line.split()
                cl = float(parts[4])
                cd = float(parts[9])
                return torch.tensor([cl, cd], dtype=torch.float32)
        except Exception as e:
            print(f"Error parsing {polar_path}: {e}")
            return torch.tensor([0.0, 0.0], dtype=torch.float32)

    def __getitem__(self, index):
        paths = self.file_pairs[index]

        # --- A. 处理点云 (三路采样合并) ---
        pc_raw = np.load(paths['pc_path'])
        # 严格按照设定的点数取值
        pc_uni = pc_raw['uniform'][:self.num_points_uniform]
        pc_cur = pc_raw['curvature'][:self.num_points_curvature]
        pc_imp = pc_raw['importance'][:self.num_points_importance]
        
        # 拼接成 [12000, 3]
        pc_input = np.concatenate([pc_uni, pc_cur, pc_imp], axis=0)

        # --- B. 处理 SDF 采样 (表面重采样逻辑) ---
        sdf_data = np.load(paths['sdf_path'])
        vol_points = sdf_data['vol_points']
        vol_sdf = sdf_data['vol_sdf']
        near_points = sdf_data['near_points']
        near_sdf = sdf_data['near_sdf']
        
        if 'surface_points' in sdf_data:
            surface_points = sdf_data['surface_points']
            surface_sdf = np.zeros((surface_points.shape[0], 1), dtype=near_sdf.dtype)
            if near_sdf.ndim == 1: near_sdf = near_sdf[:, None]
            if vol_sdf.ndim == 1: vol_sdf = vol_sdf[:, None]
            detail_points = np.concatenate([near_points, surface_points], axis=0)
            detail_sdf = np.concatenate([near_sdf, surface_sdf], axis=0)
        else:
            detail_points = near_points
            detail_sdf = near_sdf
            if detail_sdf.ndim == 1: detail_sdf = detail_sdf[:, None]
            if vol_sdf.ndim == 1: vol_sdf = vol_sdf[:, None]

        # 计算采样数量
        num_surface = int(self.num_points_sdf * self.surface_ratio)
        num_volume = self.num_points_sdf - num_surface

        # 表面点随机采样
        if detail_points.shape[0] > 0:
            idx_detail = np.random.choice(detail_points.shape[0], num_surface, replace=True)
            sampled_detail_points = detail_points[idx_detail]
            sampled_detail_values = detail_sdf[idx_detail]
        else:
            sampled_detail_points = np.zeros((num_surface, 3), dtype=np.float32)
            sampled_detail_values = np.zeros((num_surface, 1), dtype=np.float32)

        if vol_points.shape[0] > 0:
            idx_vol = np.random.choice(vol_points.shape[0], num_volume, replace=True)
            sampled_vol_points = vol_points[idx_vol]
            sampled_vol_values = vol_sdf[idx_vol]
        else:
            sampled_vol_points = np.zeros((num_volume, 3), dtype=np.float32)
            sampled_vol_values = np.zeros((num_volume, 1), dtype=np.float32)

        sdf_points_sampled = np.concatenate([sampled_detail_points, sampled_vol_points], axis=0)
        sdf_values_sampled = np.concatenate([sampled_detail_values, sampled_vol_values], axis=0)

        # --- C. 解析气动标签 ---
        aero_label = self._parse_polar(paths['polar_path'])

        return {
            'point_cloud': torch.from_numpy(pc_input).float(),
            'sdf_points': torch.from_numpy(sdf_points_sampled).float(),
            'sdf_values': torch.from_numpy(sdf_values_sampled).float(),
            'aero_label': aero_label,
            'file_id': paths['file_id']
        }

    def __len__(self):
        return len(self.file_pairs)