import os
from pathlib import Path
from box import process # 确保你的 box 模块中有对应的 process 函数

def batch_process_meshes(input_root, output_root):
    """
    批量将 input_root 下所有 mesh 处理为 SDF npz 文件
    """
    # 建立输出目录
    out_dir = os.path.join(output_root, 'sdf_data')
    os.makedirs(out_dir, exist_ok=True)

    # 遍历所有文件
    for dirpath, dirnames, filenames in os.walk(input_root):
        for fname in filenames:
            # 匹配 G*_*.obj 格式
            if fname.endswith('.obj') and fname.startswith('G'):
                mesh_path = os.path.join(dirpath, fname)
                
                # --- 逻辑调整区 ---
                # 方案 A: 如果你想让输出文件名和输入完全一样 (G2_1.npz)
                fid = os.path.splitext(fname)[0] 
                
                # 方案 B: 如果你只想保留 G2_1 这种结构，并加上 _mesh 后缀 (G2_1_mesh.npz)
                # fid = f"{os.path.splitext(fname)[0]}_mesh"
                # -----------------

                out_path = os.path.join(out_dir, f"{fid}.npz")
                
                print(f"Processing: {fname} -> {fid}.npz")
                
                try:
                    process(mesh_path, out_path)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

if __name__ == '__main__':
    # 建议使用绝对路径或确保路径在 Linux 环境下正确
    input_root = r"/home/yuwenshi/B737/B737_500"
    output_root = r"/home/yuwenshi/B737/B737_sdf"

    batch_process_meshes(input_root, output_root)