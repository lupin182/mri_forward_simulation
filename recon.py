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

def reconstruct_image_multi(
    k_space_signal, 
    k_traj_adc, 
    n_slices: int,          # 【必填】总层数
    fov_x=220e-3, 
    fov_y=220e-3, 
    Nx=64, 
    Ny=64
):
    """
    多层MRI图像重建（严格适配 拼接式输入维度）

    - k_space_signal:  (N_samples_total,)  复数K空间信号（所有层拼接）
    - k_traj_adc:      (3, N_samples_total) K空间轨迹（X/Y/Z, 所有层拼接）
    - N_samples_total = n_slices * n_samples_per_slice

    输出：
    - image_3d: (n_slices, Ny, Nx)  3D多层图像体积
    """
    print("===== 多层MRI图像重建（拼接式输入）=====")
    n_samples_per_slice = Nx * Ny
    # 数据格式标准化
    k_space_signal = np.asarray(k_space_signal, dtype=np.complex128)
    k_traj_adc = np.asarray(k_traj_adc)
    total_samples = n_slices * n_samples_per_slice

    # 输入校验
    assert k_space_signal.shape[0] == total_samples, "信号总采样数不匹配！"
    assert k_traj_adc.shape == (3, total_samples), "轨迹维度必须为 (3, 总采样数)！"

    # ===================== 核心：按层切分拼接数据 =====================
    # 1. 切分 K空间信号：(total,) → [层1, 层2, ..., 层N]
    signal_slices = np.split(k_space_signal, n_slices)
    
    # 2. 切分 K空间轨迹：(3, total) → n_slices × (3, n_samples_per_slice)
    traj_slices = np.split(k_traj_adc, n_slices, axis=1)
    # =================================================================
    import sigpy as sp
    image_stack = []
    # 逐层独立重建
    for i in range(n_slices):
        print(f"正在重建第 {i+1}/{n_slices} 层...")
        sig = signal_slices[i]       # 当前层信号 (n_samples_per_slice,)
        traj = traj_slices[i]        # 当前层轨迹 (3, n_samples_per_slice)

        # 👇 完全保留你原版的物理正确逻辑
        # 1. 提取XY平面，丢弃Z轴
        k_locs = traj[:2, :].T
        # 2. 物理单位转换 (cycles/m → 矩阵索引)
        k_locs[:, 0] *= fov_x
        k_locs[:, 1] *= fov_y
        # 3. 坐标系对齐 (x,y) → (y,x)
        k_locs_sigpy = k_locs[:, ::-1]
        # 4. NUFFT重建
        img = sp.nufft_adjoint(sig, k_locs_sigpy, (Ny, Nx))
        image_stack.append(img)

    # 堆叠为3D体积
    image_3d = np.array(image_stack)
    print(f"✅ 重建完成！输出形状: {image_3d.shape} (层数, Ny, Nx)")
    return image_3d


def reconstruct_image_3d(k_space_signal, k_traj_adc, fov_x=220e-3, fov_y=220e-3, fov_z=(3e-3)*3, Nx=64, Ny=64, Nz=3):
    """
    使用 NUFFT 将一维的 K 空间信号和 3D 轨迹重建为 3D 图像体积 (Volume)
    
    参数:
    - k_space_signal: 形状为 (N_samples,) 的复数数组 (你的 Bloch 主循环输出)
    - k_traj_adc: 形状为 (3, N_samples) 的坐标数组 (来自 seq.calculate_kspace())
    - fov_x, fov_y, fov_z: 模拟时设置的 3D 物理视野 (单位: 米)
    - Nx, Ny, Nz: 期望重建的 3D 图像矩阵大小
    """
    print("开始 3D NUFFT 重建图像...")
    import sigpy as sp
    # 强制转换为 Numpy 复数数组 (防范 list 报错)
    k_space_signal = np.asarray(k_space_signal, dtype=np.complex64)
    if k_space_signal.ndim > 1 and k_space_signal.shape[0] == 1:
        k_space_signal = k_space_signal.flatten()

    # 1. 提取完整的 3D 轨迹坐标
    # k_traj_adc 的形状是 (3, N_samples)，我们提取前三行 (即 X, Y, Z 坐标)
    k_locs = k_traj_adc[:3, :].T  # 转置后形状变为 (N_samples, 3)
    
    # 2. 物理单位转换 (cycles/meter 转换为矩阵的 index 单位)
    k_locs[:, 0] *= fov_x  # X 轴坐标转换 (kx)
    k_locs[:, 1] *= fov_y  # Y 轴坐标转换 (ky)
    k_locs[:, 2] *= fov_z  # Z 轴坐标转换 (kz)
    
    # 3. 对齐算法库的坐标系习惯 
    # Numpy 和 SigPy 期望的 3D 矩阵顺序通常是 [z, y, x] (即 [深度, 高度, 宽度])
    # k_locs 当前的顺序是 [x, y, z]，使用 ::-1 刚好将其完美翻转为 [z, y, x]
    k_locs_sigpy = k_locs[:, ::-1] 
    
    # 4. 定义 3D 目标图像的大小
    img_shape = (Nz, Ny, Nx)
    
    # 5. 调用 NUFFT 的伴随算子 (Adjoint NUFFT)
    # SigPy 会自动识别 k_locs_sigpy 的维度(3列)，并执行 3D 网格化和 3D-IFFT
    image_3d = sp.nufft_adjoint(k_space_signal, k_locs_sigpy, img_shape)
    
    print("3D 重建完成！")
    return image_3d
