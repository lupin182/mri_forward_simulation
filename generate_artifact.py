
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
