import numpy as np
import matplotlib.pyplot as plt
from utils import make_sinc, make_trap, compress_channel, add_block


# 计算不同切片层所需要的频率偏移
def get_slice_z_position_freq(slice_idx, Nz, slice_thickness, dense_rf_amp, dense_gz, GAMMA=42.576e6):
    """
    参数:
        slice_idx: 当前切片索引 (0 到 Nz-1)
        Nz: 切片总数
        slice_thickness: 切片厚度 (单位: 米)
    """
    # 1. 找到“中间索引” (Center Index)
    # 例如: 4层 -> 中间是 1.5; 5层 -> 中间是 2.0
    center_index = (Nz - 1) / 2.0
    
    # 2. 计算相对于中心的偏移量
    offset_steps = slice_idx - center_index
    
    # 3. 乘以厚度得到物理距离
    z_pos = offset_steps * slice_thickness
    
    raw_freq_line = dense_gz * GAMMA * z_pos
    is_rf_on = np.abs(dense_rf_amp) > 1e-9
    dense_freq = raw_freq_line * is_rf_on
    return dense_freq


def generate_dense_spin_echo_sequence_modified(
        phase_index_idx,  # 当前相位编码索引 (0 ~ Ny-1)
        dt,               # 仿真步长 (s)
        TE,               # 回波时间 (s)
        TR,               # 重复时间 (s)
        Ny,               # 相位编码总步数
        rf_dur,           # RF脉冲平顶时长 (s)
        gx_flat_dur,      # 读出梯度平顶时长 (s)
        ramp_time,        # 梯度爬坡时间 (s)
        pe_dur,           # 相位/重聚梯度的平顶时长 (s), 对应外部计算幅度的 Gy_dur
        Gz_amp,           # 切片梯度幅度
        Gx_amp,           # 读出梯度幅度
        Gy_max_amp,        # 最大相位编码梯度幅度
        TxCoilmg,         # 发射线圈
        TxCoilpe,         # 发射线圈
        slice_idx
    ):
    """
    生成符合物理原理的自旋回波(Spin Echo)稠密序列
    """
    
    # 初始化序列容器
    # RF 变成 List of Lists: [[coil1_data...], [coil2_data...], ...]
    TxCoilNum = TxCoilmg.shape[0]
    rf_amp = [[] for _ in range(TxCoilNum)]
    rf_phase = [[] for _ in range(TxCoilNum)]

    gx, gy, gz = [], [], []
    adc = []

    TxCoilmg = TxCoilmg[:,slice_idx, :, :]
    TxCoilpe = TxCoilpe[:,slice_idx, :, :]

    # ==========================================
    # --- Step 1: 90度 激发 (Slice Selection) ---
    # ==========================================
    # 构造波形
    rf_90, rf_phase_90 = make_sinc(75, rf_dur, dt, TxCoilmg, TxCoilpe, mask=None)
    gz_90 = make_trap(Gz_amp, rf_dur, ramp_time, dt)

    # 确定 Step 1 的总时长
    len_step1 = max(len(rf_90[0]), len(gz_90)) * dt
    
    # 对齐 RF 到梯度平顶中心
    # 假设 make_trap 生成是: [ramp_up, flat, ramp_down]
    # 90度脉冲的物理中心时刻 (相对于 Step 1 开始)
    t_center_90 = ramp_time + rf_dur / 2.0
    
    start_idx = int(ramp_time/dt)
    end_idx = start_idx + int(rf_dur/dt)
    rf_mask = np.zeros([TxCoilNum, len(gz_90)])
    rf_phase_mask = np.zeros([TxCoilNum, len(gz_90)])
    # 简单的边界保护，防止 float 转 int 误差
    actual_len = min(len(rf_90[0]), end_idx - start_idx)
    rf_mask[:, start_idx : start_idx + actual_len] = rf_90[:, :actual_len]
    rf_phase_mask[:, start_idx : start_idx + actual_len] = rf_phase_90[:, :actual_len]

    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, len_step1, rfa=rf_mask, rfp=rf_phase_mask, gzv=gz_90)

    # =======================================================
    # --- Step 2: 联合梯度段 (Gx预相位 + Gy编码 + Gz重聚) ---
    # =======================================================
    # 这一段通常很短，我们统一使用传入的 pe_dur 作为平顶时间
    t_step2_flat = pe_dur 
    #t_step2_total_dur = t_step2_flat + 2 * ramp_time # 梯形总长

    # 1. Gz 重聚 (Slice Rephasing)
    # 目的：抵消 90度 脉冲后半段造成的散相
    # 需抵消的面积 = Gz_amp * (rf_dur/2 + ramp_time * 0.5) 
    # *注意：这里假设梯度是理想梯形，且自旋在 ramp 期间也受影响
    area_gz_half = Gz_amp * (rf_dur * 0.5 + ramp_time * 0.5)
    gz_reph_amp = -area_gz_half / (t_step2_flat + ramp_time) # Area = Amp * (Flat + Ramp)
    gz_reph = make_trap(gz_reph_amp, t_step2_flat, ramp_time, dt)

    # 2. Gx 预相位 (Readout Pre-phasing)
    # 目的：使回波峰值出现在读出梯度的中心
    # 需抵消的面积 = 读出梯度前半段面积 = Gx_amp * (gx_flat_dur/2 + ramp_time * 0.5)
    area_gx_half = Gx_amp * (gx_flat_dur * 0.5 + ramp_time * 0.5)
    gx_pre_amp = -area_gx_half / (t_step2_flat + ramp_time)
    gx_pre = make_trap(gx_pre_amp, t_step2_flat, ramp_time, dt)

    # 3. Gy 相位编码 (Phase Encoding)
    # 使用传入的 pe_dur，确保与外部 Gy_max_amp 计算时的假设一致
    current_ky_idx = phase_index_idx - Ny/2
    # 线性映射: -Gy_max ... +Gy_max
    current_gy_amp = Gy_max_amp * (current_ky_idx / (Ny/2))
    gy_enc = make_trap(current_gy_amp, t_step2_flat, ramp_time, dt)

    # 添加 Block
    max_len_step2 = max(len(gz_reph), len(gx_pre), len(gy_enc)) * dt
    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, max_len_step2, gxv=gx_pre, gyv=gy_enc, gzv=gz_reph)

    # ==========================================
    # --- Step 3: 死区 1 (Wait for 180 Pulse) ---
    # ==========================================
    # 
    # 目标: 180脉冲的“中心”必须位于 90脉冲“中心” + TE/2 处
    
    current_time_abs = len(rf_amp[0]) * dt
    
    # 180度脉冲块(Step 4)的中心相对于该块起点的偏移量
    offset_center_180_block = ramp_time + rf_dur / 2.0
    
    # 目标：Step 4 应该在什么时候开始？
    # Start_Step4 + offset = t_center_90 + TE/2
    target_start_step4 = (t_center_90 + TE / 2.0) - offset_center_180_block
    
    delay1 = target_start_step4 - current_time_abs

    if delay1 < 0:
        # 这是一个非常好的调试信息，告诉你为什么 TE 设置失败
        raise ValueError(f"TE太短! 至少需要: {(current_time_abs + offset_center_180_block - t_center_90)*2000:.2f}ms")
    
    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, delay1)

    # ==========================================
    # --- Step 4: 180度 重聚脉冲 ---
    # ==========================================
    rf_180, rf_phase_180 = make_sinc(180, rf_dur, dt, TxCoilmg, TxCoilpe)

    gz_180 = make_trap(Gz_amp, rf_dur, ramp_time, dt) # 180度通常也伴随选层梯度，或者是 Crushers
    
    len_step4 = max(len(rf_180), len(gz_180)) * dt
    
    start_idx = int(ramp_time/dt)
    end_idx = start_idx + int(rf_dur/dt)

    rf_mask = np.zeros([TxCoilNum, len(gz_180)])
    rf_phase_mask = np.zeros([TxCoilNum, len(gz_180)])
    actual_len = min(len(rf_180[0]), end_idx - start_idx)
    rf_mask[:, start_idx : start_idx+actual_len] = rf_180[:, :actual_len]
    rf_phase_mask[:, start_idx : start_idx+actual_len] = rf_phase_180[:, :actual_len]

    # 注意 rfp=90 (CPMG条件通常是90度相位，或者与90脉冲相差90度)
    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, len_step4, rfa=rf_mask, rfp=np.pi/2+rf_phase_mask, gzv=gz_180)

    # ==========================================
    # --- Step 5: 死区 2 (Wait for Echo) ---
    # ==========================================
    # 目标: 回波中心 (Readout Center) 必须位于 90脉冲“中心” + TE 处
    # 或者简单来说，是对称的：Delay2 应该等于 Delay1 (如果是理想对称序列)
    # 但为了稳健，我们再次用绝对时间计算
    
    current_time_abs = len(rf_amp[0]) * dt
    
    # Step 6 (读出) 的中心相对于 Step 6 起点的偏移
    offset_center_readout_block = ramp_time + gx_flat_dur / 2.0
    
    # 目标：Step 6 应该在什么时候开始？
    # Start_Step6 + offset = t_center_90 + TE
    target_start_step6 = (t_center_90 + TE) - offset_center_readout_block
    
    delay2 = target_start_step6 - current_time_abs
    
    if delay2 < 0:
        raise ValueError("计算出的 Delay2 小于 0，检查读出梯度时长是否过长。")
        
    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, delay2)

    # ==========================================
    # --- Step 6: 读出 (Readout / ADC) ---
    # ==========================================
    gx_read = make_trap(Gx_amp, gx_flat_dur, ramp_time, dt)
    
    # 构建 ADC 掩码 (仅在平顶期间采集)
    adc_mask = np.zeros(len(gx_read))
    start_idx = int(ramp_time/dt)
    end_idx = start_idx + int(gx_flat_dur/dt)
    # 边界保护
    end_idx = min(end_idx, len(adc_mask))
    adc_mask[start_idx:end_idx] = 1
    
    add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, len(gx_read)*dt, gxv=gx_read, adcv=adc_mask)

    # ==========================================
    # --- Step 7: 填充剩余 TR ---
    # ==========================================
    current_total_time = len(rf_amp[0]) * dt
    rest_of_tr = TR - current_total_time
    
    # 调试打印 (可选)
    # print(f"Slice {phase_index_idx}: Seq Len={current_total_time*1000:.1f}ms, Rest={rest_of_tr*1000:.1f}ms")
    
    if rest_of_tr < 0:
        raise ValueError(f"TR太短! 序列长 {current_total_time*1000:.1f}ms > TR {TR*1000}ms")
    elif rest_of_tr > 0:
        add_block(rf_amp, rf_phase, gx, gy, gz, adc, dt, rest_of_tr)

    return np.array(rf_amp), np.array(rf_phase), np.array(gz), np.array(gy), np.array(gx), np.array(adc)

    
# 生成一个TR内的VSeq
def generate_spin_echo_VSeq(dt, TE, TR, FOV_x, FOV_y, Nz, Nx, Ny, SliceThick, slice_idx, phase_line,
                rf_dur, gx_flat_dur, ramp_time, Gy_dur, TxCoilmg, TxCoilpe, GAMMA=42.576e6, dense_ext=None, return_dict=True):

    # ==========================================
    # 梯度幅度计算
    # ==========================================
    rf_bw = 4.0 / rf_dur # 4.0是时间-带宽积 (TBW)
    Gz_amp = rf_bw / (GAMMA * SliceThick)
    Gx_amp = Nx / (GAMMA * FOV_x * gx_flat_dur) # 读出梯度
    Gy_max_amp = (Ny / 2) / (FOV_y * GAMMA * Gy_dur) # 最大相位梯度

    dense_rf, dense_phase, dense_gz, dense_gy, dense_gx, dense_adc = generate_dense_spin_echo_sequence_modified(phase_line, dt, TE, TR, Ny, rf_dur, gx_flat_dur, ramp_time, Gy_dur,
                                                                                              Gz_amp, Gx_amp, Gy_max_amp, TxCoilmg, TxCoilpe, slice_idx)

    # 只有RF有幅值时，相位才有意义
    is_rf_on = np.abs(dense_rf) > 1e-9
    dense_phase *= is_rf_on

    # Ext拓展，暂时全为0
    if dense_ext == None:
        dense_ext = np.zeros_like(dense_gx)

    # 频率偏移
    dense_freq = get_slice_z_position_freq(slice_idx, Nz, SliceThick, dense_rf, dense_gz)

    # 生成flag和Line
    tsLine = []

    flag_rf, rfAmpLine, tsLine = compress_channel(dense_rf, tsLine)
    flag_phase, rfPhaseLine, tsLine = compress_channel(dense_phase, tsLine)
    flag_freq, rfFreqLine, tsLine = compress_channel(dense_freq, tsLine)

    flag_gz, GzAmpLine, tsLine = compress_channel(dense_gz, tsLine)

    flag_gy, GyAmpLine, tsLine = compress_channel(dense_gy, tsLine)
    flag_gx, GxAmpLine, tsLine = compress_channel(dense_gx, tsLine)
    flag_adc, _, tsLine = compress_channel(dense_adc, tsLine)
    flag_ext, ExtLine, tsLine = compress_channel(dense_ext, tsLine)

    utsLine = np.arange(0, len(dense_rf[0])*dt, dt) 

    '''
    # ADC截断,如果保留此块,模拟时间会变短,但只能模拟出T2加权图像
    one_indices = np.where(dense_adc == 1)[0]
    if len(one_indices) != 0:
        # 最后一次出现 1 的位置
        last_one_idx = one_indices[-1]

        # 截断位置：最后一个 1 后面再加 5
        cut_idx = last_one_idx + 100

        # 防止越界
        cut_idx = min(cut_idx, len(utsLine))

        # 截断 utsLine（保留到 cut_idx，不包含之后）
        utsLine = utsLine[:cut_idx]
    '''

    tsLine = np.unique(np.array(tsLine))
    flagsLine = np.column_stack((flag_rf, flag_phase, flag_freq, flag_gz, flag_gy, flag_gx, flag_adc, flag_ext))
    flagsLine = flagsLine[(flagsLine != 0).any(axis=1)]

    # ==========================================
    # 8. 绘图验证 (验证稠密数据即可，因为那是物理真值)
    # ==========================================
    '''
    plt.figure(figsize=(10, 8))
    plt.subplot(5,1,1); plt.plot(dense_rf[0,:5000]); plt.title('RF (Dense)')
    plt.subplot(5,1,2); plt.plot(dense_gz[:5000]); plt.title('Gz (Dense)')
    plt.subplot(5,1,3); plt.plot(dense_gx[:5000]); plt.title('Gx (Dense)')
    plt.subplot(5,1,4); plt.plot(dense_gy[:5000]); plt.title('Gy (Dense)')
    plt.subplot(5,1,5); plt.plot(dense_adc[:5000]); plt.title('ADC (Dense)')
    plt.tight_layout()
    plt.show()
    '''
    if return_dict:
        VSeq = {
            'utsLine': utsLine,
            'tsLine': tsLine,
            'flagsLine': flagsLine,
            'rfAmpLine': rfAmpLine,
            'rfPhaseLine': rfPhaseLine,
            'rfFreqLine': rfFreqLine,
            'GzAmpLine': GzAmpLine,
            'GyAmpLine': GyAmpLine,
            'GxAmpLine': GxAmpLine,
            'ExtLine': ExtLine # 无插件调用
        }
        return VSeq
    else:
        return utsLine,tsLine,flagsLine,rfAmpLine,rfPhaseLine,rfFreqLine,GzAmpLine,GyAmpLine,GxAmpLine,ExtLine
    
    
if __name__ == "__main__":
    # ==========================================
    # 1. 物理常数与参数 (保持不变)
    # ==========================================
    GAMMA = 42.576e6  
    dt = 10e-6 # 单位：秒
    TE = 20e-3 # 单位：秒
    TR = 1000e-3 # 单位：秒
    FOV_x = 0.20 # 单位：米
    FOV_y = 0.20 # 单位：米
    Nz = 3
    Nx = 64 # 单位：像素
    Ny = 64 # 单位：像素
    SliceThick = 0.005 # 单位：米
    slice_idx = 1
    rf_dur = 2e-3 # 单位：秒
    Gy_dur = 2e-4
    gx_flat_dur = 4e-3 # 单位：秒
    ramp_time = 0.5e-3 # 单位：秒
    phase_line = 33

    VSeq = generate_spin_echo_VSeq(dt, TE, TR, FOV_x, FOV_y, Nz, Nx, Ny, SliceThick, slice_idx, phase_line,
                rf_dur, gx_flat_dur, Gy_dur, ramp_time)

