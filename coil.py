from phantom.make_phantom import generate_diff_coil_sensitivity_maps
import matplotlib.pyplot as plt
import numpy as np

def plot_4coil_sensitivity(coil_mag, coil_phase):
    """
    专门绘制【4个线圈】的MRI接收线圈灵敏度分布图
    展示内容：每个线圈的 灵敏度幅值 + 相位分布
    布局：2行 × 4列（第一行：幅值，第二行：相位）
    """
    # 固定绘制4个线圈，自动截取前4个通道
    n_coils_plot = 4
    coil_names = ["Coil1", "Coil2", "Coil3", "Coil4"]
    
    # 创建画布：2行(幅值/相位) × 4列(4个线圈)
    fig, axes = plt.subplots(2, n_coils_plot, figsize=(16, 8))
    #fig.suptitle('MRI 4通道接收线圈灵敏度分布图', fontsize=18, y=0.95)

    for c in range(n_coils_plot):
        # 提取2D数据（原数据维度 [n_coil, 1, Nx, Ny]，去掉Nz维度）
        mag = coil_mag[c, 0, :, :]
        phase = coil_phase[c, 0, :, :]

        # ---------- 绘制灵敏度幅值（第一行）----------
        im_mag = axes[0, c].imshow(
            mag.T,        # 转置匹配图像坐标
            cmap='viridis',
            origin='lower'# 左下为坐标原点
        )
        axes[0, c].set_title(f'{coil_names[c]}\nAmplitude', fontsize=12)
        axes[0, c].axis('off')
        plt.colorbar(im_mag, ax=axes[0, c], shrink=0.7)

        # ---------- 绘制灵敏度相位（第二行）----------
        im_phase = axes[1, c].imshow(
            phase.T,
            cmap='hsv',   # 环形色图，最适合相位(0~2π)
            vmin=0, vmax=2*np.pi,
            origin='lower'
        )
        axes[1, c].set_title(f'{coil_names[c]}\nPhase', fontsize=12)
        axes[1, c].axis('off')
        plt.colorbar(im_phase, ax=axes[1, c], shrink=0.7)

    plt.tight_layout()
    plt.show()

# ====================== 测试：生成并绘制4线圈灵敏度 ======================
if __name__ == "__main__":
    # 生成4个线圈的灵敏度数据
    coil_mag, coil_phase, _ = generate_diff_coil_sensitivity_maps(n_coils=4)
    # 绘制分布图
    plot_4coil_sensitivity(coil_mag, coil_phase)
    
    image=np.load('image_recon_coil.npy')
    plt.figure(figsize=(20, 20))
    plt.subplot(1,4,1)
    plt.title("Coil 1")
    plt.imshow(np.abs(image[1,0]), cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Coil 2")
    plt.imshow(np.abs(image[3,0]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.title("Coil 3")
    plt.imshow(np.abs(image[0,0]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.title("Coil 4")
    plt.imshow(np.abs(image[2,0]), cmap='gray')
    plt.axis('off')
    plt.show()

    from recon import sos_reconstruction
    image_recon = sos_reconstruction(image)
    plt.figure(figsize=(20, 20))
    plt.title("SOS Reconstruction")
    plt.imshow(np.abs(image_recon[0]), cmap='gray')
    plt.axis('off')
    plt.show()
