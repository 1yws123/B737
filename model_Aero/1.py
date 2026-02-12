from dataset import SDFDataset
dataset = SDFDataset(
    pc_root_dir='/home/yuwenshi/B737/G58_pc_1299/pointcloud',
    aero_root_dir='/home/yuwenshi/B737/G58_aero_1299/G58_aero_1299',
    sdf_dir='/home/yuwenshi/B737/G58_sdf_1299/sdf_data'
)
print(f"第一个样本的数据摘要: {dataset[2]['aero_label']}")