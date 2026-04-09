import numpy as np


def create_chemical_shift_phantom(
    Nz: int = 1,
    Nx: int = 64,
    Ny: int = 64,
    B0: float = 3.0   # 3.0T主磁场，化学位移Hz单位
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ✅ 严格按照MRI仿真维度定义：(TypeNum, SpinNum, Nz, Nx, Ny)
    ✅ TypeNum=1：每个体素仅1种组织
    ✅ 化学位移单位：Hz
    ✅ 体膜空间分布：左脂肪 + 右水（频率编码X轴，完美产生化学位移伪影）
    
    返回：pdmap, t1map, t2map, csmap  维度均为 (1,1,Nz,Nx,Ny)
    """
    # 强制单组织单自旋体素
    TypeNum = 1
    SpinNum = 1
    # 初始化数组：严格匹配维度 (TypeNum, SpinNum, Nz, Nx, Ny)
    pdmap = np.zeros((TypeNum, SpinNum, Nz, Nx, Ny), dtype=np.float32)
    t1map = np.zeros((TypeNum, SpinNum, Nz, Nx, Ny), dtype=np.float32)
    t2map = np.zeros((TypeNum, SpinNum, Nz, Nx, Ny), dtype=np.float32)
    csmap = np.zeros((TypeNum, SpinNum, Nz, Nx, Ny), dtype=np.float32)

    # 旋磁比：H质子 γ/2π = 42.58 MHz/T = 42.58 Hz/(ppm·T)
    fat_cs_ppm = -3.5 #-3.4
    fat_cs_hz = B0 * 42.576 * fat_cs_ppm  # 脂肪化学位移(Hz)
    water_cs_hz = 0.0                    # 水化学位移(Hz)

    # ===================== 空间分块：左脂肪、右水 =====================
    half_x = Nx // 2  # X轴中点，水脂分界线

    # 1. 右侧体素（X >= half_x）：水组织参数
    pdmap[0, 0, :, half_x:, :] = 1.0
    t1map[0, 0, :, half_x:, :] = 3.0  # s
    t2map[0, 0, :, half_x:, :] = 0.3   # s
    csmap[0, 0, :, half_x:, :] = water_cs_hz

    # 2. 左侧体素（X < half_x）：脂肪组织参数
    pdmap[0, 0, :, :half_x, :] = 0.9
    t1map[0, 0, :, :half_x:, :] = 0.25  # s
    t2map[0, 0, :, :half_x:, :] = 0.08  # s
    csmap[0, 0, :, :half_x, :] = fat_cs_hz

    return pdmap, t1map, t2map, csmap

if __name__ == '__main__':
    from phantom.make_phantom import Phantom
    from Sequence.write_gre_label import write_gre_label_sequence
    from simulate import simulate, SimulationConfig
    from recon import reconstruct_3d_cartesian_fft
    import matplotlib.pyplot as plt
    pdmap, t1map, t2map, csmap = create_chemical_shift_phantom(Nx=64, Ny=64)
    phantom = Phantom(pdmap, t1map, t2map, fov_x=0.512, fov_y=0.512, slice_thickness=5e-3, CS=csmap)
    seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
                                fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
                                tr=100e-3, te=20e-3)

    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)

    plt.figure(figsize=(10, 10))
    plt.subplot(1,2,1)
    plt.title("Reconstruction")
    plt.imshow(np.abs(image_recon[0]), cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Original")
    plt.imshow(pdmap[0, 0, 0],  cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
