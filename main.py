"""Project entry point for a Cartesian GRE forward-simulation demo."""

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom, load_simple_phantom
from recon import reconstruct_image_fft, reconstruct_image, reconstruct_image_multi, reconstruct_image_3d
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi import write_epi_sequence
from Sequence.write_se import write_se_sequence
from Sequence.write_epi_se_rs import write_epi_se_rs_sequence
from Sequence.write_epi_label import write_epi_label_sequence
from Sequence.write_gre import write_gre_sequence
from simulate import SimulationConfig, simulate

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


def main() -> None:
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)
    rho, t1, t2 = load_simple_phantom("E:\毕业课题\old_version\mrisimulation_test\output\discrete_phantom_3.0T_miao.nii", 90)
    '''
    rho = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').pixel_array
    t1 = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_T1Map_301_aligned/00000012.dcm').pixel_array
    t2 = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_T2Star_307_aligned/00000012.dcm').pixel_array
    rho = rho.astype(np.float32)
    t1 = t1.astype(np.float32)
    t2 = t2.astype(np.float32)
    rho = rho[np.newaxis, np.newaxis, np.newaxis, :, :]
    t1 = t1[np.newaxis, np.newaxis, np.newaxis, :, :]/1000
    t2 = t2[np.newaxis, np.newaxis, np.newaxis, :, :]/1000
    
    Nz, Nx, Ny = rho.shape[2:]
    dx = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').PixelSpacing[0]/1000
    dy = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').PixelSpacing[1]/1000
    '''
    FOV_x =  0.181# 单位：米
    FOV_y = 0.217 # 单位：米

    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=0.001)
    # The current forward model uses one isochromat per voxel, so RF spoiling
    # creates stronger artifacts than a scanner would. Disable it for the demo.
    seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
                            fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
                            tr=100e-3,te=20e-3)

    #seq = write_epi_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
     #                       fov=(phantom.fov_x, phantom.fov_y), 
      #                      n_slices=1)

    k_traj_adc,_,_,_,_ = seq.calculate_kspace()
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))

    #image_recon, _ = reconstruct_image_fft(k_space_signal, Ny=64, Nx=64)
    #image_recon = reconstruct_image_multi(k_space_signal, k_traj_adc, 
    #                                        n_slices=phantom.Nz, Nx=phantom.Nx, Ny=phantom.Ny)
    image_recon, _ = reconstruct_image_fft(k_space_signal, Ny=phantom.Ny, Nx=phantom.Nx)
    #ans = plot_color_overlay(np.abs(image_recon_2[0]),rho[0, 0, 0])
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Reconstruction")
    plt.imshow(np.abs(image_recon), cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title("Original")
    plt.imshow(rho[0, 0, 0],  cmap='gray')
    plt.axis('off')


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
