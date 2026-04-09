from device_manager import device_manager
import numpy as np

def generate_rf_artifact(t_adc, k_space_signal, 
                        rf_noise_freq=[0.0], rf_noise_amp=[0.0], bg_noise_amp=0.0,
                        B0=3, Gamma=42.576e6, bandwidth=50e3):
    """
    生成RF伪影,包括指定频率的射频噪声和高斯背景噪声
    输入:
        t_adc: 采集ADC信号的时间点,与K空间信号一一对应,是一维数组形状为(N,)
        k_space_signal: K空间信号,是一维数组,与t_adc一一对应,形状为(N,)
        rf_noise_freq: 射频噪声频率列表
        rf_noise_amp: 射频噪声振幅列表
        bg_noise_amp: 背景噪声振幅
    返回:
        k_space_with_artifact: 添加了RF伪影的K空间信号
    """

    # 复制原始信号以避免修改输入
    k_space_with_artifact = np.copy(k_space_signal)
    
    # 添加指定频率的射频噪声
    for freq, amp in zip(rf_noise_freq, rf_noise_amp):
        # 生成正弦波噪声（实部和虚部分别添加）
        demodulated_freq = freq - Gamma * B0
        if abs(demodulated_freq) > bandwidth:
            # 频率超出带宽范围,不添加
            continue
        rf_noise = amp * np.exp(1j * 2 * np.pi * demodulated_freq * t_adc)
        k_space_with_artifact += rf_noise
    
    # 添加高斯背景噪声
    if bg_noise_amp > 0:
        # 实部和虚部分别添加高斯噪声
        noise_real = np.random.normal(0, bg_noise_amp, size=k_space_signal.shape)
        noise_imag = np.random.normal(0, bg_noise_amp, size=k_space_signal.shape)
        bg_noise = noise_real + 1j * noise_imag
        k_space_with_artifact += bg_noise
    
    return k_space_with_artifact


from phantom.make_phantom import Phantom
def generate_B0_inhomogeneity(phantom:Phantom, mode: str, delta_B0_ppm: float, axis: str = 'x'):
    """
    生成3T主磁场不均匀性场图，输出单位：特斯拉（T）
    支持线性分布 / 抛物线分布两种建模方式
    
    参数
    ----------
    mode : str
        分布模式：'linear'（线性）/ 'parabolic'（抛物线/碗状）
    delta_B0_ppm : float
        主磁场不均匀度（ppm），例：0.5 → 0.5ppm
    axis : str
        线性模式的分布轴：'x' / 'y'，仅线性模式生效
    
    返回
    ----------
    B0_map : np.ndarray
        主磁场不均匀场，形状 (Nz, Nx, Ny)，单位：特斯拉（T）
    """
    # 主磁场强度 3T
    B0_nominal = phantom.B0
    # ppm → 特斯拉 核心换算（1 ppm = 1e-6）
    ppm_to_T = B0_nominal * (delta_B0_ppm / 1e6)

    # ===================== 线性分布（沿X/Y轴）=====================
    if mode == "linear":
        if axis not in ["x", "y"]:
            raise ValueError("线性模式仅支持 x / y 轴")
        
        # 选取对应轴的中心化坐标
        coord = phantom.x if axis == "x" else phantom.y
        # 计算FOV总长度，归一化到 [-0.5, 0.5]
        fov_length = phantom.dx * phantom.Nx if axis == "x" else phantom.dy * phantom.Ny
        normalized_coord = coord / fov_length  
        # 输出：特斯拉（T）
        B0_map = ppm_to_T * normalized_coord

    # ===================== 抛物线分布（碗状，X-Y平面）=====================
    elif mode == "parabolic":
        # 到中心的径向距离平方（x²+y²）
        r_square = phantom.x ** 2 + phantom.y ** 2
        # 归一化分母（FOV对角最大值）
        half_fov_x = (phantom.Nx / 2) * phantom.dx
        half_fov_y = (phantom.Ny / 2) * phantom.dy
        norm_denominator = half_fov_x ** 2 + half_fov_y ** 2

        # 输出：特斯拉（T）
        B0_map = ppm_to_T * (r_square / norm_denominator)

    else:
        raise ValueError("模式仅支持 linear / parabolic")

    # 保存到类属性并返回
    phantom.dB0 = device_manager.to_device(B0_map)
    return B0_map