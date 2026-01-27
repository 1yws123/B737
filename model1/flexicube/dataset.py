import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import re

class SDFDataset(Dataset):
    def __init__(self, pc_root_dir, sdf_dir, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000,  # 新增
                 num_points_importance=4000,
                 num_points_sdf=16384, surface_ratio=0.8,surface_threshold=0.02):
        
        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature # 新增
        self.num_points_importance = num_points_importance
        
        # 总点数现在是三者之和
        self.num_points_pc = num_points_uniform + num_points_curvature + num_points_importance
        
        self.num_points_sdf = num_points_sdf
        self.surface_ratio = surface_ratio
        
        self.file_pairs = self._make_dataset(pc_root_dir, sdf_dir)
        if not self.file_pairs:
            raise RuntimeError("无法配對數據")
        print(f"Dataset 初始化: {len(self.file_pairs)} 樣本")

    def _make_dataset(self, pc_root_dir, sdf_dir):
        # ... (保持原有的配对逻辑不变) ...
        pc_map = {}
        # 确保路径匹配你的实际情况
        pc_files = glob.glob(os.path.join(pc_root_dir, 'G*_*_pc.npz'))
             
        for pc_path in pc_files:
            match = re.search(r'(G\d+_\d+)_pc\.npz', os.path.basename(pc_path))
            if match: pc_map[match.group(1)] = pc_path
        
        file_pairs = []
        sdf_files = glob.glob(os.path.join(sdf_dir, '*.npz'))
        for sdf_path in sdf_files:
            match = re.search(r'(G\d+_\d+)\.npz', os.path.basename(sdf_path))
            if match:
                fid = match.group(1)
                if fid in pc_map:
                    file_pairs.append((pc_map[fid], sdf_path))
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def _resample_points(self, points, target_num):
        """
        辅助函数：将点云重采样到指定数量 (多退少补)
        """
        num_curr = points.shape[0]
        if num_curr == 0:
            # 如果该类型没有点，填充零 (或者考虑随机噪声)
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

    def __getitem__(self, idx):
        pc_path, sdf_path = self.file_pairs[idx]
        
        # --- 1. 加载输入点雲 (Point Cloud) ---
        try:
            pc_data = np.load(pc_path)
            
            # 读取原始数据 (使用你提供的安全读取逻辑)
            u_pts = pc_data['uniform'] if 'uniform' in pc_data else np.zeros((0, 3))
            c_pts = pc_data['curvature'] if 'curvature' in pc_data else np.zeros((0, 3)) # 新增
            i_pts = pc_data['importance'] if 'importance' in pc_data else np.zeros((0, 3))

            # 分别重采样到固定数量
            # 这一步至关重要，它保证了拼接后的顺序结构 [Uniform, Curvature, Importance]
            u_pts_fixed = self._resample_points(u_pts, self.num_points_uniform)
            c_pts_fixed = self._resample_points(c_pts, self.num_points_curvature) # 新增
            i_pts_fixed = self._resample_points(i_pts, self.num_points_importance)

            # 拼接
            pc_raw = np.concatenate([u_pts_fixed, c_pts_fixed, i_pts_fixed], axis=0)
            
            # Jitter 增强
            jitter = np.clip(0.005 * np.random.randn(*pc_raw.shape), -0.01, 0.01).astype(np.float32)
            pc_raw += jitter
            
        except Exception as e:
            print(f"Error loading PC {pc_path}: {e}")
            # 返回全0以防崩溃
            pc_raw = np.zeros((self.num_points_pc, 3), dtype=np.float32)

        # --- 2. 加载 GT SDF 数据 ---
        sdf_data = np.load(sdf_path)
        normalization_shift = sdf_data['shifts'].astype(np.float32)
        normalization_scale = sdf_data['scale'].astype(np.float32)

        vol_points = sdf_data['vol_points']
        vol_sdf = sdf_data['vol_sdf']
        near_points = sdf_data['near_points']
        near_sdf = sdf_data['near_sdf']
        
        # 兼容有无 surface_points 的情况
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

        # SDF 采样逻辑
        num_detail = int(self.num_points_sdf * self.surface_ratio)
        num_vol = self.num_points_sdf - num_detail
        
        if detail_points.shape[0] > 0:
            idx_detail = np.random.choice(detail_points.shape[0], num_detail, replace=True)
            sampled_detail_points = detail_points[idx_detail]
            sampled_detail_values = detail_sdf[idx_detail]
        else:
            sampled_detail_points = np.zeros((num_detail, 3), dtype=np.float32)
            sampled_detail_values = np.zeros((num_detail, 1), dtype=np.float32)

        if vol_points.shape[0] > 0:
            idx_vol = np.random.choice(vol_points.shape[0], num_vol, replace=True)
            sampled_vol_points = vol_points[idx_vol]
            sampled_vol_values = vol_sdf[idx_vol]
        else:
            sampled_vol_points = np.zeros((num_vol, 3), dtype=np.float32)
            sampled_vol_values = np.zeros((num_vol, 1), dtype=np.float32)

        sdf_points_sampled = np.concatenate([sampled_detail_points, sampled_vol_points], axis=0)
        sdf_values_sampled = np.concatenate([sampled_detail_values, sampled_vol_values], axis=0)

        return {
            'point_cloud': torch.from_numpy(pc_raw).float(),
            'sdf_points': torch.from_numpy(sdf_points_sampled).float(),
            'sdf_values': torch.from_numpy(sdf_values_sampled).float(),
            'shift': torch.from_numpy(normalization_shift).float(),
            'scale': torch.from_numpy(np.array(normalization_scale)).float()
        }