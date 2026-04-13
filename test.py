
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_display_images():
    """
    读取test_picture文件夹中不同ppm伪影的图片矩阵以及groundtruth，
    用matplotlib画图显示
    """
    print("="*80)
    print("读取并显示MRI伪影图像")
    print("="*80)
    
    # 定义ppm值和对应的文件路径
    ppm_values = [20, 30, 35]
    base_path = "test_picture"
    
    # 存储所有图像数据
    images = {}
    
    # 读取所有图像
    print("\n[步骤1] 读取图像文件...")
    for ppm in ppm_values:
        # 读取理想图像（ground truth）
        ideal_path = os.path.join(base_path, f"image_ideal_db0_{ppm}ppm.npy")
        if os.path.exists(ideal_path):
            img_ideal = np.load(ideal_path)
            images[f"ideal_{ppm}ppm"] = img_ideal
            print(f"  已读取: {ideal_path}")
        
        # 读取有伪影图像
        artifact_path = os.path.join(base_path, f"image_artifact_db0_{ppm}ppm.npy")
        if os.path.exists(artifact_path):
            img_artifact = np.load(artifact_path)
            images[f"artifact_{ppm}ppm"] = img_artifact
            print(f"  已读取: {artifact_path}")
    
    print(f"\n  共读取 {len(images)} 张图像")
    
    # ==================== 生成综合对比图 - Ground Truth + 不同ppm伪影 ====================
    print("\n[步骤2] 生成综合对比图 (Ground Truth + 不同ppm伪影)...")
    
    # 使用30ppm作为ground truth参考
    ref_ppm = 30
    img_gt = np.abs(np.squeeze(images[f"ideal_{ref_ppm}ppm"]))
    
    fig = plt.figure(figsize=(18, 10))
    
    # 第一行：Ground Truth
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(img_gt, cmap='gray')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 第一行后三个位置留空或显示空图
    for i in range(2, 5):
        ax = plt.subplot(2, 4, i)
        ax.axis('off')
    
    # 第二行：不同ppm的伪影图像
    for i, ppm in enumerate(ppm_values):
        ax = plt.subplot(2, 4, 4 + i + 1)
        img_artifact = np.abs(np.squeeze(images[f"artifact_{ppm}ppm"]))
        im = ax.imshow(img_artifact, cmap='gray')
        ax.set_title(f'With {ppm} ppm Artifact', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('ppm_artifact_comparison_v1.png', dpi=300, bbox_inches='tight')
    print("  综合对比图v1已保存为: ppm_artifact_comparison_v1.png")
    plt.close()
    
    # ==================== 生成另一种布局：每行是一个ppm值的完整对比 ====================
    print("\n[步骤3] 生成详细对比图 (每行一个ppm值的完整对比)...")
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, ppm in enumerate(ppm_values):
        # 该行的第一个子图：Ground Truth
        ax1 = plt.subplot(3, 4, i*4 + 1)
        img_gt_ppm = np.abs(np.squeeze(images[f"ideal_{ppm}ppm"]))
        im1 = ax1.imshow(img_gt_ppm, cmap='gray')
        ax1.set_title(f'Ground Truth ({ppm} ppm)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 第二个子图：有伪影图像
        ax2 = plt.subplot(3, 4, i*4 + 2)
        img_artifact = np.abs(np.squeeze(images[f"artifact_{ppm}ppm"]))
        im2 = ax2.imshow(img_artifact, cmap='gray')
        ax2.set_title(f'With Artifact ({ppm} ppm)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 第三个子图：差异图像
        ax3 = plt.subplot(3, 4, i*4 + 3)
        diff_img = img_artifact - img_gt_ppm
        vmax = np.max(np.abs(diff_img))
        im3 = ax3.imshow(diff_img, cmap='seismic', vmin=-vmax, vmax=vmax)
        ax3.set_title(f'Difference ({ppm} ppm)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 第四个子图：剖面图
        ax4 = plt.subplot(3, 4, i*4 + 4)
        center_y = img_gt_ppm.shape[0] // 2
        profile_gt = img_gt_ppm[center_y, :]
        profile_artifact = img_artifact[center_y, :]
        x = np.arange(len(profile_gt))
        ax4.plot(x, profile_gt, 'b-', linewidth=2, label='Ground Truth')
        ax4.plot(x, profile_artifact, 'r--', linewidth=2, label=f'With {ppm} ppm Artifact')
        ax4.set_title(f'Center Horizontal Profile', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Pixel')
        ax4.set_ylabel('Magnitude')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppm_artifact_comparison_v2.png', dpi=300, bbox_inches='tight')
    print("  详细对比图v2已保存为: ppm_artifact_comparison_v2.png")
    plt.close()
    
    # ==================== 生成第三种布局：Ground Truth在左上角，其他是伪影 ====================
    print("\n[步骤4] 生成网格对比图 (Ground Truth + 所有伪影)...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 使用一个通用的ground truth
    img_gt_common = np.abs(np.squeeze(images[f"ideal_{ref_ppm}ppm"]))
    
    # 显示Ground Truth
    ax_gt = plt.subplot(2, 2, 1)
    im_gt = ax_gt.imshow(img_gt_common, cmap='gray')
    ax_gt.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
    
    # 显示20ppm伪影
    ax_20 = plt.subplot(2, 2, 2)
    img_20 = np.abs(np.squeeze(images[f"artifact_20ppm"]))
    im_20 = ax_20.imshow(img_20, cmap='gray')
    ax_20.set_title('20 ppm Artifact', fontsize=16, fontweight='bold')
    ax_20.axis('off')
    plt.colorbar(im_20, ax=ax_20, fraction=0.046, pad=0.04)
    
    # 显示30ppm伪影
    ax_30 = plt.subplot(2, 2, 3)
    img_30 = np.abs(np.squeeze(images[f"artifact_30ppm"]))
    im_30 = ax_30.imshow(img_30, cmap='gray')
    ax_30.set_title('30 ppm Artifact', fontsize=16, fontweight='bold')
    ax_30.axis('off')
    plt.colorbar(im_30, ax=ax_30, fraction=0.046, pad=0.04)
    
    # 显示35ppm伪影
    ax_35 = plt.subplot(2, 2, 4)
    img_35 = np.abs(np.squeeze(images[f"artifact_35ppm"]))
    im_35 = ax_35.imshow(img_35, cmap='gray')
    ax_35.set_title('35 ppm Artifact', fontsize=16, fontweight='bold')
    ax_35.axis('off')
    plt.colorbar(im_35, ax=ax_35, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('ppm_artifact_comparison_v3.png', dpi=300, bbox_inches='tight')
    print("  网格对比图v3已保存为: ppm_artifact_comparison_v3.png")
    plt.close()
    
    print("\n" + "="*80)
    print("图像处理完成！")
    print("="*80)
    print("\n生成的图像文件:")
    print("  1. ppm_artifact_comparison_v1.png (GT在第一行，伪影在第二行)")
    print("  2. ppm_artifact_comparison_v2.png (每行一个ppm值的完整对比)")
    print("  3. ppm_artifact_comparison_v3.png (网格布局：GT + 三个伪影)")
    
    return images

if __name__ == '__main__':
    images = load_and_display_images()
