from .device_manager import device_manager
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
    k_space_with_artifact = np.copy(k_space_signal).squeeze()
    
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

import numpy as np

def generate_rf_artifact_real(t_adc, k_space_signal, 
                        rf_noise_freq=[0.0], rf_noise_amp=[0.0], bg_noise_amp=0.0,
                        B0=3, Gamma=42.576e6, bandwidth=50e3,
                        # 新增：真实噪声可调参数（默认值即可满足真实感，无需修改）
                        phase_randomize=True,    # 随机初始相位（核心改进）
                        freq_drift_ratio=5,   # 频率微漂移比例（模拟干扰源不稳定）
                        amp_mod_depth=2         # 幅度慢调制深度（模拟信号波动）
                        ):
    """
    🔥 改进版：生成真实磁共振射频伪影（非相干窄带干扰+热噪声）
    解决：纯单频相干噪声导致的「离散高亮像素+条纹发黑」问题
    输入:
        t_adc: 采集ADC信号的时间点,与K空间信号一一对应,一维数组 (N,)
        k_space_signal: K空间信号,一维数组 (N,)
        rf_noise_freq: 射频噪声频率列表 [Hz]
        rf_noise_amp: 射频噪声振幅列表
        bg_noise_amp: 高斯背景热噪声振幅
        B0: 主磁场强度 (T)
        Gamma: 旋磁比 (Hz/T)
        bandwidth: 接收带宽 (Hz)
        phase_randomize: 开启随机相位（核心！破坏相干干涉）
        freq_drift_ratio: 频率微漂移比例
        amp_mod_depth: 幅度慢调制深度
    返回:
        k_space_with_artifact: 添加真实RF伪影的K空间信号
    """
    # 保留原始信号维度，避免维度错误
    k_space_with_artifact = np.copy(k_space_signal).astype(np.complex128)
    N_samples = len(t_adc)
    
    # ===================== 核心改进：真实射频窄带噪声 =====================
    for freq, amp in zip(rf_noise_freq, rf_noise_amp):
        demod_freq = freq - Gamma * B0

        print(f"原始频率 {freq} -> 解调后频率 {demod_freq}")
        # 带宽过滤（保留原逻辑）
        if abs(demod_freq) > bandwidth:
            print(f"频率 {freq} 超出带宽 {bandwidth}，不添加")
            continue
            
        # 1. 🔥 频率微漂移（模拟真实干扰源频率不稳定，展宽频谱）
        drift = demod_freq * freq_drift_ratio * np.random.uniform(-1, 1)
        real_freq = demod_freq + drift
        
        # 2. 🔥 随机初始相位（核心！破坏固定相位相干干涉）
        if phase_randomize:
            init_phase = np.random.uniform(0, 2 * np.pi)
        else:
            init_phase = 0.0
            
        # 3. 🔥 幅度慢调制（模拟射频干扰强度缓慢波动，非恒定值）
        time_normalized = t_adc - t_adc[0]
        amp_mod = 1.0 - amp_mod_depth * np.sin(2 * np.pi * 0.1 * time_normalized)
        time_varying_amp = amp * amp_mod
        
        # 4. 生成真实非相干射频噪声
        rf_noise = time_varying_amp * np.exp(1j * (2 * np.pi * real_freq * t_adc + init_phase))
        
        # 叠加到k空间
        k_space_with_artifact += rf_noise
    
    # ===================== 改进：标准MRI接收热噪声 =====================
    if bg_noise_amp > 0:
        # 生成标准复高斯噪声（MRI接收机标准噪声模型）
        bg_noise = np.random.normal(0, bg_noise_amp, k_space_signal.shape) + \
                   1j * np.random.normal(0, bg_noise_amp, k_space_signal.shape)
        k_space_with_artifact += bg_noise
    
    return k_space_with_artifact.squeeze()


from .phantom import Phantom
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
