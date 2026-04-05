import numpy as np
import matplotlib.pyplot as plt

def reconstruct_image(k_space_signal, k_traj_adc, fov_x=220e-3, fov_y=220e-3, Nx=64, Ny=64):
    """
    使用 NUFFT 将一维的 K 空间信号和轨迹重建为 2D 图像
    
    参数:
    - k_space_signal: 形状为 (N_samples,) 的复数数组 (你的 Bloch 主循环输出)
    - k_traj_adc: 形状为 (N_dims, N_samples) 的坐标数组 (来自 seq.calculate_kspace())
    - fov_x, fov_y: 模拟时设置的物理视野 (单位: 米)
    - Nx, Ny: 期望重建的图像矩阵大小 (如 64, 64)
    """
    print("开始重建图像...")
    
    # 1. 提取 2D 轨迹坐标 (抛弃 Z 轴，如果是纯 2D 扫描)
    # k_traj_adc 的形状是 (3, N_samples)，我们只取前两行 X 和 Y
    k_locs = k_traj_adc[:2, :].T  # 转置后形状变为 (N_samples, 2)
    
    # 2. 极其关键的物理单位转换！
    # PyPulseq 计算出的 k_traj_adc 单位是绝对物理单位：cycles / meter (每米多少个循环)
    # 但 SigPy 等算法库要求坐标被归一化到 [-N/2, N/2] 的图像矩阵索引单位。
    # 转换公式极其简单：cycles/meter * meters (FOV) = cycles (即矩阵索引)
    k_locs[:, 0] *= fov_x  # X 轴坐标转换
    k_locs[:, 1] *= fov_y  # Y 轴坐标转换
    
    # 3. 对齐算法库的坐标系习惯 (SigPy 期望的顺序通常是 [ky, kx])
    # 将 (x, y) 翻转为 (y, x) 以匹配图像矩阵 (Ny, Nx)
    k_locs_sigpy = k_locs[:, ::-1] 
    
    # 4. 定义目标图像的大小
    img_shape = (Ny, Nx)
    
    # 5. 密度补偿 (Density Compensation)
    # 如果你的轨迹中心密集、边缘稀疏（如螺旋轨迹或放射状），必须做这一步来消除模糊。
    # 对于标准的笛卡尔 EPI/GRE，不做也行，但做了图像更锐利。
    # 这里我们使用一个简单的 Voronoi 密度补偿函数 (如果有需要的话)
    # dcf = sp.mri.radial_dcf(k_locs_sigpy, img_shape) # 仅作演示，视轨迹而定
    
    # 6. 调用 NUFFT 的伴随算子 (Adjoint NUFFT)
    # 这相当于先在 K 空间把散乱的点“网格化(Gridding)”到标准的 64x64 网格上，然后再做 2D-IFFT
    import sigpy as sp

    image = sp.nufft_adjoint(k_space_signal, k_locs_sigpy, img_shape)
    
    print("重建完成！")
    return image

def reconstruct_image_fft(k_space_signal, Ny, Nx):
    """
    使用基础 2D-IFFT 将均匀的 K 空间信号重建为图像
    
    参数:
    - k_space_signal: 一维复数数组 (你的 Bloch 主循环输出的信号)
    - Nx: 频率编码(读出)方向的采样点数 (通常等于 block.adc.num_samples)
    - Ny: 相位编码的步数 (层内的 TR 次数或 EPI 的 blip 次数)
    """
    print("开始基础 FFT 重建...")
    
    # 1. 强制转换为 Numpy 复数数组 (防止传入的是 list)
    k_signal_array = np.asarray(k_space_signal, dtype=np.complex64)
    
    # 2. 形状检查
    expected_len = Nx * Ny
    if k_signal_array.size != expected_len:
        raise ValueError(f"信号长度不匹配！期望 {expected_len}，实际 {k_signal_array.size}。\n"
                         f"请检查是否包含多层数据，或有没有多余的采样点。")
    
    # 3. 将 1D 信号折叠成 2D 的 K 空间矩阵
    # 根据扫描习惯，通常外层循环是相位编码(Ny)，内层是读出(Nx)
    k_space_2d = k_signal_array.reshape((Ny, Nx))
    
    # ==========================================
    # 核心数学变换：标准 MRI IFFT 流程
    # ==========================================
    
    # 第一步：ifftshift - 把 K 空间的中心低频部分移到矩阵的四个角上
    # （因为标准的 FFT 算法认为原点 [0,0] 应该在左上角）
    k_shifted = np.fft.ifftshift(k_space_2d)
    
    # 第二步：ifft2 - 执行二维快速傅里叶逆变换
    img_complex = np.fft.ifft2(k_shifted)
    
    # 第三步：fftshift - 把重建出的图像空间中心平移回矩阵的正中央
    # （否则图像会被撕裂成四块分布在四个角）
    image = np.fft.fftshift(img_complex)
    
    print("重建完成！")
    return image, k_space_2d
