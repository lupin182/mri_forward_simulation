import numpy as np
import matplotlib.pyplot as plt


def plot_color_overlay(img1, img2, title="Color Overlay"):
    """
    生成两张图像的伪彩叠加图 (Color Overlay)
    红色 = 仅存在于图 1 的信号
    绿色 = 仅存在于图 2 的信号
    黄色 = 完美重合
    """
    print("正在生成伪彩叠加图...")
    
    # 1. 强制取模（防范 MRI 重建出的复数矩阵）
    img1_mag = np.abs(img1)
    img2_mag = np.abs(img2)
    
    # 2. 形状对齐检查
    if img1_mag.shape != img2_mag.shape:
        raise ValueError(f"图像尺寸不匹配！图 1 为 {img1_mag.shape}，图 2 为 {img2_mag.shape}。\n"
                         f"请确保它们的分辨率和 FOV 已经对齐。")
                         
    # 3. 归一化到 [0, 1] 区间
    # 极其重要！必须归一化，否则因为信号强度的绝对数值不同，根本调和不出黄色
    img1_norm = (img1_mag - np.min(img1_mag)) / (np.max(img1_mag) - np.min(img1_mag) + 1e-8)
    img2_norm = (img2_mag - np.min(img2_mag)) / (np.max(img2_mag) - np.min(img2_mag) + 1e-8)
    
    # 4. 组装 RGB 三通道矩阵
    # 形状为 (Ny, Nx, 3)
    rgb_composite = np.zeros((*img1_norm.shape, 3))
    
    # 图 1 塞进红色通道 (R)
    rgb_composite[..., 0] = img1_norm  
    # 图 2 塞进绿色通道 (G)
    rgb_composite[..., 1] = img2_norm  
    # 蓝色通道 (B) 保持为 0
    
    # 5. 绘图展示
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb_composite)
    
    # 添加图例说明
    plt.title(f"{title}\n(Red = Img 1, Green = Img 2, Yellow = Match)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return rgb_composite


def reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Nx, Ny, Nz):
    """
    基于绝对坐标的 3D 笛卡尔 K 空间 FFT 重建
    
    参数:
    - k_space_signal: shape为 (N,) 的复数数组 (一维的采样信号)
    - k_traj_adc: shape为 (3, N) 的坐标数组 (物理单位，如 cycles/m)
    - Nx, Ny, Nz: 目标重建矩阵的三维尺寸
    
    返回:
    - image_3d: 经过 3D IFFT 重建出的复数图像矩阵，shape为 (Nz, Ny, Nx)
    - k_space_3d: 重组后标准化的 3D K空间矩阵 (可用于检查数据填充是否正确)
    """
    print("开始基于坐标映射的 3D 笛卡尔 FFT 重建...")
    
    k_signal = np.asarray(k_space_signal, dtype=np.complex64)
    
    # ==========================================
    # 1. 坐标归一化：将物理的连续坐标转化为离散的矩阵索引
    # ==========================================
    def coords_to_indices(coords, grid_size):
        """将一维坐标数组映射到 [0, grid_size - 1] 的整数索引区间"""
        c_min, c_max = np.min(coords), np.max(coords)
        
        # 应对 2D 扫描的情况 (比如 Nz=1，所有 Z 坐标都为 0)
        if c_max == c_min:
            return np.zeros_like(coords, dtype=int)
            
        # 线性映射到 0 ~ N-1
        normalized = (coords - c_min) / (c_max - c_min)
        indices = np.round(normalized * (grid_size - 1)).astype(int)
        
        # 裁剪以防极其微小的浮点误差越界
        return np.clip(indices, 0, grid_size - 1)

    # 提取 X(读出), Y(相位), Z(选层) 的坐标并转化为整数索引
    ix = coords_to_indices(k_traj_adc[0, :], Nx)
    iy = coords_to_indices(k_traj_adc[1, :], Ny)
    iz = coords_to_indices(k_traj_adc[2, :], Nz)
    
    # ==========================================
    # 2. 三维 K 空间网格填充 (Gridding)
    # ==========================================
    # 创建一个全零的标准 3D K 空间矩阵 (Z, Y, X)
    k_space_3d = np.zeros((Nz, Ny, Nx), dtype=np.complex64)
    
    # 使用 Numpy 的高级索引将一维数据精准填入对应坐标
    # 这种做法极其强大！即使是 EPI 反向采样的行，因为坐标是正确的，它也会自动反着填入矩阵！
    k_space_3d[iz, iy, ix] = k_signal
    
    # ==========================================
    # 3. 核心数学变换：3D-IFFT
    # ==========================================
    # 第一步：移频 (原点移到角落)
    k_shifted = np.fft.ifftshift(k_space_3d)
    
    # 第二步：三维傅里叶逆变换
    img_complex = np.fft.ifftn(k_shifted)
    
    # 第三步：中心移回
    image_3d = np.fft.fftshift(img_complex)
    
    print("3D 重建完成！")
    return image_3d, k_space_3d