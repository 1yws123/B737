# model_sdf.py (修改版)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# 1. & 2. 辅助函数和模块 (保持不变)
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn(xyz, k):
    B, N, _ = xyz.shape
    dist = torch.cdist(xyz, xyz)
    idx = dist.topk(k=k, largest=False)[1]
    return idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            grouped_points = grouped_points.permute(0, 2, 1).unsqueeze(2)
        else:
            if self.npoint is None or self.npoint == 0:
                new_xyz = xyz
            else:
                new_xyz_idx = farthest_point_sample(xyz, self.npoint)
                new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint or N, 1, 3)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        if self.group_all:
            new_points = torch.max(grouped_points, 3)[0]
        else:
            new_points = torch.max(grouped_points, 2)[0]

        return new_xyz, new_points.permute(0, 2, 1)

class GeometryAwareAttentionBlock(nn.Module):
    def __init__(self, in_channels, k=16):
        super(GeometryAwareAttentionBlock, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.linear_mhsa = nn.Linear(in_channels, in_channels)
        self.linear_knn1 = nn.Linear(in_channels, in_channels)
        self.linear_knn2 = nn.Linear(in_channels, in_channels)
        self.linear_concat = nn.Linear(in_channels * 2, in_channels)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, xyz, features):
        B, N, C = features.shape
        attn_output, _ = self.mhsa(features, features, features)
        global_features = self.linear_mhsa(attn_output)
        knn_idx = knn(xyz, k=self.k)
        knn_features = index_points(features, knn_idx)
        processed_knn_features = self.relu(self.linear_knn1(knn_features))
        local_features = torch.max(processed_knn_features, dim=2)[0]
        local_features = self.linear_knn2(local_features)
        concatenated_features = torch.cat([global_features, local_features], dim=-1)
        fused_features = self.relu(self.linear_concat(concatenated_features))
        output_features = self.norm1(fused_features + features)
        return output_features    

class HashEmbedder(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = n_levels * n_features_per_level

        # 计算每一层的分辨率缩放因子 b
        # b = exp( (ln(finest) - ln(base)) / (L - 1) )
        self.b = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList()
        for i in range(n_levels):
            resolution = math.floor(base_resolution * (self.b ** i))
            
            # 决定是用 Dense Grid 还是 Hash Grid
            # 如果所需参数量 < hashmap_size，就用密集网格 (Dense)，否则用哈希表
            n_params_dense = resolution ** 3
            if n_params_dense < (2 ** log2_hashmap_size):
                num_entries = n_params_dense
            else:
                num_entries = 2 ** log2_hashmap_size
            
            # 初始化 Embedding Table
            embedding = nn.Embedding(num_entries, n_features_per_level)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)

        # 空间哈希的素数，用于解决冲突
        self.primes = [1, 2654435761, 805459861]

    def forward(self, x):
        # x shape: [batch, num_points, 3], range 应该在 [0, 1] 之间
        # 如果你的输入范围是 [-1, 1]，请先归一化到 [0, 1]: x = (x + 1) / 2
        
        batch_size, num_points, _ = x.shape
        x_flat = x.view(-1, 3)
        
        outputs = []
        for i in range(self.n_levels):
            resolution = math.floor(self.base_resolution * (self.b ** i))
            
            # 1. 缩放坐标到当前分辨率
            x_scaled = x_flat * resolution
            
            # 2. 找到左下角整数坐标 (x0) 和右上角 (x1)
            x0 = torch.floor(x_scaled).long()
            x1 = x0 + 1
            
            # 3. 计算用于三线性插值的权重 (fractional part)
            weights = x_scaled - x0.float() # [N, 3]
            
            # 4. 获取 8 个角的坐标
            # corners shape: [8, N, 3]
            # 000, 001, 010, 011, 100, 101, 110, 111
            corners = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corners.append(x0 + torch.tensor([dx, dy, dz], device=x.device))
            
            # 5. 哈希索引或线性索引计算
            embedding_layer = self.embeddings[i]
            num_entries = embedding_layer.num_embeddings
            
            corner_features = []
            for corner in corners:
                # 空间哈希逻辑
                if num_entries < (resolution + 1) ** 3: # 使用 Hash
                    hashed_indices = (
                        (corner[:, 0] * self.primes[0]) ^ 
                        (corner[:, 1] * self.primes[1]) ^ 
                        (corner[:, 2] * self.primes[2])
                    ) % num_entries
                else: # 使用 Dense Grid 线性索引
                    # 需要限制边界防止越界（虽然 x0, x1 理论上在范围内，但为了安全）
                    # 注意：这里简化了 Dense Indexing，假设输入是严格 [0,1]
                    hashed_indices = (
                        corner[:, 0] * (resolution + 1) * (resolution + 1) +
                        corner[:, 1] * (resolution + 1) +
                        corner[:, 2]
                    ) % num_entries
                
                corner_features.append(embedding_layer(hashed_indices))
            
            # 6. 三线性插值 (Trilinear Interpolation)
            # weights: [N, 3] -> wx, wy, wz
            wx, wy, wz = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
            
            # 这里的插值逻辑：
            # c000(0), c001(1), c010(2), c011(3), c100(4), c101(5), c110(6), c111(7)
            c00 = corner_features[0] * (1-wx) + corner_features[4] * wx
            c01 = corner_features[1] * (1-wx) + corner_features[5] * wx
            c10 = corner_features[2] * (1-wx) + corner_features[6] * wx
            c11 = corner_features[3] * (1-wx) + corner_features[7] * wx
            
            c0 = c00 * (1-wy) + c10 * wy
            c1 = c01 * (1-wy) + c11 * wy
            
            c = c0 * (1-wz) + c1 * wz
            
            outputs.append(c)
            
        # 拼接所有层级的特征
        return torch.cat(outputs, dim=-1).view(batch_size, num_points, -1)
    
# ===============================================================
# 3. Encoder (核心修改部分)
# ===============================================================
class Encoder(nn.Module):
    # 3.1: 新增参数来接收两种点云的数量
    def __init__(self, latent_dim=128, num_fourier_freqs=8, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, # 新增
                 num_points_importance=4000):
        super(Encoder, self).__init__()
        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature # 新增
        self.num_points_importance = num_points_importance

        self.input_embedder = FourierEmbedder(num_freqs=num_fourier_freqs, input_dim=3)
        sa1_in_channel = self.input_embedder.out_dim
        
        # 3.2: 【核心修改】将单一的 sa1 替换为两个独立的 SA 模块
        # 它们拥有各自独立的权重，将分别学习处理不同类型的几何信息。
        # 每个模块负责从 4096 点下采样到 256 点。
        self.sa1_uniform = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        # 2. Curvature 分支 (新增)
        self.sa1_curvature = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        # 3. Importance 分支
        self.sa1_importance = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)

        # 后续层保持不变
        self.geo_attn = GeometryAwareAttentionBlock(in_channels=128, k=16)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # sa2 的输入点数现在是 1024+1024+1024=3072
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, xyz):
        # xyz 的 shape 是 [B, 8192, 3]

        # 3.3: 【核心修改】分割输入的点云和特征
        # 对应 Hunyuan 代码中的 torch.split
        idx_u_end = self.num_points_uniform
        idx_c_end = self.num_points_uniform + self.num_points_curvature
        
        # 1. 切分坐标
        xyz_uniform = xyz[:, :idx_u_end, :]
        xyz_curvature = xyz[:, idx_u_end:idx_c_end, :] # 中间段是 Curvature
        xyz_importance = xyz[:, idx_c_end:, :]

        initial_features = self.input_embedder(xyz)
        features_uniform = initial_features[:, :idx_u_end, :]
        features_curvature = initial_features[:, idx_u_end:idx_c_end, :]
        features_importance = initial_features[:, idx_c_end:, :]

        # 3.4: 【核心修改】分别通过各自的 SA 层，实现差异化特征提取
        l1_xyz_u, l1_points_u = self.sa1_uniform(xyz_uniform, features_uniform)
        l1_xyz_c, l1_points_c = self.sa1_curvature(xyz_curvature, features_curvature)
        l1_xyz_i, l1_points_i = self.sa1_importance(xyz_importance, features_importance)

        # 3.5: 【核心修改】拼接结果，融合两种信息
        # 对应文献中的 "concatenating both sources"
        # l1_xyz 的 shape: [B, 256+256, 3] -> [B, 512, 3]
        # l1_points 的 shape: [B, 256+256, 128] -> [B, 512, 128]
        l1_xyz = torch.cat([l1_xyz_u, l1_xyz_c, l1_xyz_i], dim=1)       # [B, 768, 3]
        l1_points = torch.cat([l1_points_u, l1_points_c, l1_points_i], dim=1) # [B, 768, 128]
        
        # 5. 后续处理 (Attention -> Fusion -> SA2 -> SA3)
        l1_points_attn = self.geo_attn(l1_xyz, l1_points)
        l1_points_fused = self.fusion_conv(l1_points.transpose(1, 2) + l1_points_attn.transpose(1, 2)).transpose(1, 2)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_fused)
        _, global_feature = self.sa3(l2_xyz, l2_points)
        
        x = global_feature.squeeze(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ===============================================================
# 4. Triplane Decoder & Fourier Embedder (保持不变)
# ===============================================================
class TriplaneDecoder(nn.Module):
    def __init__(self, latent_dim=128, plane_resolution=64, plane_features=8):
        super(TriplaneDecoder, self).__init__()
        self.start_res = 4
        self.target_res = plane_resolution
        assert self.target_res >= self.start_res and (self.target_res & (self.target_res - 1) == 0), \
            f"目标分辨率 (plane_resolution) 必须是4或更高的2的幂, 但得到的是 {self.target_res}"
        num_upsamples = int(math.log2(self.target_res / self.start_res))
        self.fc_start = nn.Linear(latent_dim, 256 * self.start_res * self.start_res)
        upsample_layers = []
        in_channels = 256
        for i in range(num_upsamples):
            out_channels = 16 if i == num_upsamples - 1 else in_channels // 2
            upsample_layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        self.upsample_layers = nn.Sequential(*upsample_layers)
        self.head_xy = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        self.head_yz = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        self.head_xz = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z):
        x = self.fc_start(z)
        x = x.view(x.shape[0], 256, self.start_res, self.start_res)
        shared_features = self.upsample_layers(x)
        plane_xy = self.head_xy(shared_features)
        plane_yz = self.head_yz(shared_features)
        plane_xz = self.head_xz(shared_features)
        return plane_xy, plane_yz, plane_xz
    
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=6, input_dim=3):
        super().__init__()
        freq = 2.0 ** torch.arange(num_freqs)
        self.register_buffer("freq", freq, persistent=False)
        self.out_dim = input_dim * (num_freqs * 2 + 1)

    def forward(self, x: torch.Tensor):
        embed = (x[..., None].contiguous() * self.freq).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        
# ===============================================================
# 5. PointCloudVAE 主模型 (需要更新 __init__ 以传递参数)
# ===============================================================
class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim, plane_resolution, plane_features, num_fourier_freqs=6, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, # 新增
                 num_points_importance=4000):        # 移除 num_fourier_freqs 参数，因为它被 hash encoding 配置取代
        super(PointCloudVAE, self).__init__()
        
        self.encoder = Encoder(
            latent_dim, 
            num_fourier_freqs=num_fourier_freqs,
            num_points_uniform=num_points_uniform,
            num_points_curvature=num_points_curvature, # 新增
            num_points_importance=num_points_importance
        )
        
        self.decoder = TriplaneDecoder(
            latent_dim=latent_dim,
            plane_resolution=plane_resolution,
            plane_features=plane_features
        )
        
        # ================= 替换 FourierEmbedder 为 HashEmbedder =================
        # 配置参考 Instant-NGP 默认值: L=16, F=2, T=2^19
        self.use_hash_encoding = True
        
        if self.use_hash_encoding:
            self.pos_embedder = HashEmbedder(
                n_levels=16, 
                n_features_per_level=2, 
                log2_hashmap_size=19,
                base_resolution=16,
                finest_resolution=2048
            )
        else:
            # 保留旧选项作为对比
            self.pos_embedder = FourierEmbedder(num_freqs=6, input_dim=3)
        
        # 计算 SDF Head 的输入维度
        # Triplane features (plane_features * 3) + Positional Encoding
        input_dim_sdf_head = (plane_features * 3) + self.pos_embedder.out_dim
        
        self.sdf_head = nn.Sequential(
             nn.Linear(input_dim_sdf_head, 64), # HashEncoding 特征很强，MLP 可以稍微小一点
             nn.ReLU(),
             nn.Linear(64, 64),
             nn.ReLU(),
             nn.Linear(64, 1) # SDF 值
             # 注意：Instant-NGP 通常不在最后一层用 ReLU，因为 SDF 可以是负数
        )
        # ==============================================================================

    # reparameterize 和 forward 保持不变 ...
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        triplanes = self.decoder(z)
        return triplanes, mu, log_var

    def query_sdf(self, triplanes, query_points):
        # query_points shape: [B, N, 3]
        
        # ================= 改动 2: 坐标范围检查 =================
        # Hash Encoding 期望输入在 [0, 1]。假设 query_points 在 [-1, 1]
        # 如果你的数据已经在 [0, 1]，请注释掉下面这一行
        query_points_norm = (query_points + 1.0) / 2.0 
        # ======================================================

        plane_xy, plane_yz, plane_xz = triplanes
        batch_size, num_query_points, _ = query_points.shape

        # Triplane 采样逻辑 (使用原始坐标 [-1, 1] 因为 grid_sample 默认范围是 [-1, 1])
        # 注意：这里为了严谨，Triplane 的采样应该和 Hash 的采样分开处理坐标系
        # PyTorch grid_sample 使用 [-1, 1]
        grid_xy = query_points[:, :, [0, 1]].view(batch_size, num_query_points, 1, 2)
        grid_yz = query_points[:, :, [1, 2]].view(batch_size, num_query_points, 1, 2)
        grid_xz = query_points[:, :, [0, 2]].view(batch_size, num_query_points, 1, 2)
        
        features_xy = F.grid_sample(plane_xy, grid_xy, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        features_yz = F.grid_sample(plane_yz, grid_yz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        features_xz = F.grid_sample(plane_xz, grid_xz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        
        features_xy = features_xy.transpose(1, 2)
        features_yz = features_yz.transpose(1, 2)
        features_xz = features_xz.transpose(1, 2)
        
        # ================= 改动 3: 使用 Hash Embedder =================
        # 使用归一化到 [0, 1] 的坐标进行哈希编码
        pos_features = self.pos_embedder(query_points_norm)
        # ============================================================

        # 拼接 Triplane 特征 (全局) + Hash 特征 (局部高频)
        aggregated_features = torch.cat([features_xy, features_yz, features_xz, pos_features], dim=-1)
        
        predicted_sdf = self.sdf_head(aggregated_features)
        return predicted_sdf