

# 遍历文件，只对【数据集】使用切片，跳过【组】
import h5py
import numpy as np

# ===================== 1. 配置参数 =====================
# 【重要】Windows路径用 原始字符串r"" 或 双反斜杠\\，避免转义错误
file_path = r"E:\pythonproject\koma\KomaMRI.jl\examples\2.phantoms\sphere_db0.h5"

import h5py
import numpy as np


# 线性B0畸变参数
γ = 2 * np.pi * 42.58e6  # 旋磁比 (rad/s/T)
B0_strength = 1e-6    # 畸变强度（可调大/调小）
distort_axis = 'x'       # 线性方向：x=左右，y=上下

# ===================== 【核心】原地打开并修改文件 =====================
# r+ 模式：读写打开已有文件，**不创建、不复制、直接修改**
with h5py.File(file_path, "r+") as f:
    # 定位到数据集
    sample_group = f["sample"]
    dataset = sample_group["data"]  # 直接获取数据集句柄，不读取数据
    
    print(f"正在修改文件：{file_path}")
    print(f"数据形状：{dataset.shape}")

    # 生成匹配的网格坐标（用于计算线性Δw）
    H, W, _ = dataset.shape
    x_coords = np.linspace(-1, 1, W)
    y_coords = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x_coords, y_coords)

    # 计算纯线性Δw
    if distort_axis == 'x':
        linear_delta_w = γ * B0_strength * X
    else:
        linear_delta_w = γ * B0_strength * Y

    # ===================== 原地直接修改最后一维（Δω）=====================
    # 直接写入原数据集，无任何数据复制！
    dataset[..., -1] = linear_delta_w

print("✅ 原地修改完成！Δω 已替换为纯线性主磁场不均匀")