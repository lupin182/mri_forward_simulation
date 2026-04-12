'''
测试伪影专用文件
'''

from device_manager import get_xp, device_manager
import numpy as np
import matplotlib.pyplot as plt

xp = get_xp()

def generate_complex_phantom_and_dB0(Nz=1, Nx=128, Ny=128):

    """
    生成一个复杂的体模数据,用于主磁场不均匀伪影验证
    
    参数:
        Nz: 切片数
        Nx: x维度大小
        Ny: y维度大小
        SpinNum: 自旋池数
        TypeNum: 组织类型数
        dB0_amplitude: dB0不均匀的最大幅度(特斯拉)
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 密度
        T1:  (type, spin_packet, Nz, Nx, Ny) T1值
        T2:  (type, spin_packet, Nz, Nx, Ny) T2值
        dB0:  (type, spin_packet, Nz, Nx, Ny) dB0主磁场不均匀值
    """
    
    # 1. 初始化 (背景密度 0.1, T1=2s, T2=0.5s)
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.005
    
    # 2. 生成坐标网格
    _, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx) - Nx/2,  
        np.arange(Ny) - Ny/2, 
        indexing='ij'
    )
    
    # 3. 主体：中心大圆 (半径 Nx/4)
    r_main = Nx / 4
    mask_main = (x**2 + y**2) <= r_main**2
    '''
    # 4. 生成几个内部结构
    # 内环
    r_inner = Nx / 8
    mask_inner = (x**2 + y**2) <= r_inner**2
    
    # 右上角小方块
    mask_square1 = (x > Nx/8) & (x < Nx/4) & (y > Nx/8) & (y < Nx/4)
    # 左下角小方块
    mask_square2 = (x < -Nx/8) & (x > -Nx/4) & (y < -Nx/8) & (y > -Nx/4)
    '''
    # 5. 赋值
    # 主体：水 (Rho=1.0, T1=1.0, T2=0.1)
    rho[mask_main] = 1.0
    t1[mask_main]  = 1.0
    t2[mask_main]  = 0.1
    '''
    # 内环：不同组织 (Rho=1.5, T1=0.8, T2=0.08)
    rho[mask_inner] = 1.5
    t1[mask_inner]  = 0.8
    t2[mask_inner]  = 0.08
    
    # 右上角方块：高亮区域 (Rho=2.0, T1=0.5, T2=0.05)
    rho[mask_square1] = 2.0
    t1[mask_square1]  = 0.5
    t2[mask_square1]  = 0.05
    
    # 左下角方块：另一高亮区域 (Rho=1.8, T1=0.6, T2=0.06)
    rho[mask_square2] = 1.8
    t1[mask_square2]  = 0.6
    t2[mask_square2]  = 0.06
    '''
    
    # 7. 调整维度以符合要求: (type, spin_packet, Nz, Nx, Ny)
    rho = rho[np.newaxis, np.newaxis, :, :, :]
    t1 = t1[np.newaxis, np.newaxis, :, :, :]
    t2 = t2[np.newaxis, np.newaxis, :, :, :]
    
    return rho, t1, t2


from phantom.make_phantom import Phantom
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi_se import write_epi_se_sequence
from simulate import SimulationConfig, simulate
from recon import reconstruct_3d_cartesian_fft, plot_color_overlay
from generate_artifact import generate_B0_inhomogeneity


def run_simulation_with_dB0(rho, t1, t2, FOV_x, FOV_y, slice_thickness, 
                             ideal_spoiling_reset=True, dummy_scans=0, has_dB0=False):
    """
    运行带有指定dB0的模拟
    
    参数:
        rho, t1, t2, dB0_map: 体模参数
        FOV_x, FOV_y, slice_thickness: 成像参数
        ideal_spoiling_reset: 是否开启理想spoiling
        dummy_scans: 虚拟扫描次数
    
    返回:
        image_recon: 重建的图像
    """
    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=slice_thickness)

    if has_dB0:
        generate_B0_inhomogeneity(phantom, mode="linear", delta_B0_ppm=dB0_amplitude,axis="y")

    seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
                                fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
                                tr=100e-3, te=20e-3,
                                dummy_scans=dummy_scans,
                                ideal_spoiling_reset=ideal_spoiling_reset)

    #seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
    #                        fov=(phantom.fov_x, phantom.fov_y), te=200e-3)
    
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)
    
    return image_recon

def calculate_image_metrics(img1, img2):
    """
    计算两张图像之间的差异指标
    
    参数:
        img1: 图像1 (无伪影)
        img2: 图像2 (有伪影)
    
    返回:
        metrics: 包含各项指标的字典
    """
    img1_mag = np.abs(img1)
    img2_mag = np.abs(img2)
    
    # 归一化
    img1_norm = img1_mag / (np.max(img1_mag) + 1e-8)
    img2_norm = img2_mag / (np.max(img2_mag) + 1e-8)
    
    # 计算均方误差 (MSE)
    mse = np.mean((img1_norm - img2_norm)**2)
    
    # 计算峰值信噪比 (PSNR)
    psnr = 10 * np.log10(1 / (mse + 1e-8))
    
    # 计算结构相似性指数 (SSIM) 的简化版本
    mu1 = np.mean(img1_norm)
    mu2 = np.mean(img2_norm)
    sigma1 = np.std(img1_norm)
    sigma2 = np.std(img2_norm)
    sigma12 = np.mean((img1_norm - mu1) * (img2_norm - mu2))
    
    c1 = (0.01 * 1)**2
    c2 = (0.03 * 1)**2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    
    # 计算最大差异
    max_diff = np.max(np.abs(img1_norm - img2_norm))
    
    # 计算相对误差
    relative_error = np.mean(np.abs(img1_norm - img2_norm) / (img1_norm + 1e-8))
    
    metrics = {
        'MSE': mse,
        'PSNR (dB)': psnr,
        'SSIM': ssim,
        'Max Difference': max_diff,
        'Relative Error': relative_error
    }
    
    return metrics

def visualize_results(image_ideal, image_artifact, metrics):
    """
    可视化结果，包括理想图像、有伪影图像和对比分析
    
    参数:
        image_ideal: 无伪影理想图像
        image_artifact: 有伪影图像
        metrics: 图像质量指标
    """
    # 提取2D图像
    img_ideal_2d = np.squeeze(np.abs(image_ideal))
    img_artifact_2d = np.squeeze(np.abs(image_artifact))

    # 1. 自动适配Windows/Mac/Linux中文字体
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC", "WenQuanYi Micro Hei"]
    # 2. 解决负号乱码
    plt.rcParams["axes.unicode_minus"] = False
    
    # 创建大图
    plt.figure(figsize=(22, 22))

    # 2. 理想图像 (无dB0)
    ax2 = plt.subplot(2, 2, 1)
    im2 = ax2.imshow(img_ideal_2d, cmap='gray')
    ax2.set_title('理想成像结果\n(dB0=0)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. 有伪影图像 (有dB0)
    ax3 = plt.subplot(2, 2, 2)  
    im3 = ax3.imshow(img_artifact_2d, cmap='gray')
    ax3.set_title('含主磁场不均匀伪影\n成像结果', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. 差异图像 (Artifact - Ideal)
    ax4 = plt.subplot(2, 2, 3)
    diff_img = img_artifact_2d - img_ideal_2d
    im4 = ax4.imshow(diff_img, cmap='seismic', vmin=-np.max(np.abs(diff_img)), vmax=np.max(np.abs(diff_img)))
    ax4.set_title('差异图像\n(Artifact - Ideal)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. 伪彩叠加图
    ax5 = plt.subplot(2, 2, 4)
    img1_norm = (img_ideal_2d - np.min(img_ideal_2d)) / (np.max(img_ideal_2d) - np.min(img_ideal_2d) + 1e-8)
    img2_norm = (img_artifact_2d - np.min(img_artifact_2d)) / (np.max(img_artifact_2d) - np.min(img_artifact_2d) + 1e-8)
    rgb_composite = np.zeros((*img1_norm.shape, 3))
    rgb_composite[..., 0] = img1_norm  # 理想图像 - 红色
    rgb_composite[..., 1] = img2_norm  # 有伪影 - 绿色
    ax5.imshow(rgb_composite)
    ax5.set_title('伪彩叠加图\n(红=理想, 绿=伪影, 黄=匹配)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("主磁场不均匀伪影验证完成！")
    print("="*60)
    print("\n图像质量对比指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    #print("\n可视化结果已保存为: dB0_artifact_verification.png")

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 参数设置
    Nz = 1
    Nx = 256
    Ny = 256
    FOV_x = 0.256  # 22 cm
    FOV_y = 0.256  # 22 cm
    slice_thickness = 5e-3  # 5 mm
    dB0_amplitude = 30 #单位ppm

    print("="*60)
    print("开始主磁场不均匀伪影验证")
    print("="*60)
    print(f"成像参数: Nx={Nx}, Ny={Ny}, FOV={FOV_x*100}cm×{FOV_y*100}cm")
    print(f"dB0不均匀幅度: {dB0_amplitude} ppm")

    # 1. 生成体模和dB0数据
    print("步骤1: 生成复杂体模和dB0主磁场不均匀数据...")
    rho, t1, t2 = generate_complex_phantom_and_dB0(Nz=Nz, Nx=Nx, Ny=Ny)
    print("  [OK] 体模和dB0数据生成完成")

    # 2. 模拟理想情况 (dB0=0)
    print("步骤2: 运行理想成像模拟 (dB0=0)...")
    image_ideal = run_simulation_with_dB0(
        rho, t1, t2,
        FOV_x, FOV_y, slice_thickness,
        ideal_spoiling_reset=True, dummy_scans=0, has_dB0=False
    )
    np.save("test_picture/image_ideal_db0_50ppm.npy", image_ideal)
    print("  [OK] 理想成像模拟完成")

    # 3. 模拟有dB0伪影的情况
    print("步骤3: 运行含dB0伪影的成像模拟...")
    image_artifact = run_simulation_with_dB0(
        rho, t1, t2, 
        FOV_x, FOV_y, slice_thickness,
        ideal_spoiling_reset=True, dummy_scans=0, has_dB0=True
    )
    np.save("test_picture/image_artifact_db0_50ppm.npy", image_artifact)
    print("  [OK] 含伪影成像模拟完成")

    # 4. 计算质量指标
    print("步骤4: 计算图像质量对比指标...")
    metrics = calculate_image_metrics(image_ideal, image_artifact)
    print("  [OK] 指标计算完成")

    # 5. 可视化结果
    print("步骤5: 生成可视化结果...")
    visualize_results(image_ideal, image_artifact, metrics)

