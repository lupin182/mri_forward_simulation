import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mutual_info_score
# 新增：图像相位相关配准（自动计算平移量）
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mutual_info_score
from scipy.ndimage import shift


def calculate_mri_image_d(
    npy_file: str,
    mat_file: str,
    mat_var_name: str = "image",
    show_plot: bool = True
):
    """
    专属修复：强制整数像素对齐 → 解决左偏1格问题
    """
    # ===================== 1. 加载自定义 .npy 图像 =====================
    img_custom = np.load(npy_file)
    img_custom = np.squeeze(img_custom)
    img_custom = np.abs(img_custom).astype(np.float64)

    # ===================== 2. 加载 KomaMRI 图像 =====================
    with h5py.File(mat_file, "r") as f:
        print(f"✅ .mat 文件变量：{list(f.keys())}")
        struct_data = f[mat_var_name][:]
        img_koma = struct_data['real'] + 1j * struct_data['imag']
        img_koma = np.squeeze(img_koma.T)
        img_koma = np.abs(img_koma)
        print(img_koma.shape)
    # ===================== 🔥 核心修复：强制向右平移1像素（专治左偏1格） =====================
    print("🔧 执行强制对齐：自定义图像 → 向右平移 1 个像素")
    img_custom_aligned = shift(img_custom, shift=[0, 0], mode='constant', cval=0)  # 0=上下，1=左右

    # ===================== 归一化 =====================
    def normalize(img):
        img = img - np.min(img)
        return img / (np.max(img) + 1e-8)

    img1_norm = normalize(img_custom_aligned)
    img2_norm = normalize(img_koma)

    # ===================== 量化指标 =====================
    flat1 = img1_norm.flatten()
    flat2 = img2_norm.flatten()

    mae = np.mean(np.abs(flat1 - flat2))
    mse = np.mean((flat1 - flat2) ** 2)
    psnr_val = psnr(img1_norm, img2_norm, data_range=1.0)
    ssim_val = ssim(img1_norm, img2_norm, data_range=1.0)
    mi_val = mutual_info_score((flat1 * 255).astype(int), (flat2 * 255).astype(int))

    # ===================== 打印结果 =====================
    print("=" * 60)
    print("📊 强制整数对齐后 最终对比结果")
    print("=" * 60)
    print(f"图像尺寸: {img1_norm.shape}")
    print(f"平均绝对误差 (MAE):    {mae:.6f}")
    print(f"均方误差 (MSE):        {mse:.6f}")
    print(f"峰值信噪比 (PSNR):     {psnr_val:.2f} dB")
    print(f"结构相似性 (SSIM):     {ssim_val:.6f}")
    print(f"互信息 (MI):           {mi_val:.4f}")
    print("=" * 60)

    # ===================== 可视化 =====================
    if show_plot:
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1,4,1)
        plt.imshow(normalize(img_custom), cmap='gray')
        plt.title('origin')
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(img1_norm, cmap='gray')
        plt.title('register')
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(img2_norm, cmap='gray')
        plt.title('KomaMRI')
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(np.abs(img1_norm-img2_norm), cmap='jet')
        plt.colorbar()
        plt.title('Diff')
        plt.axis('off')

        plt.show()

    return {"SSIM":ssim_val, "PSNR":psnr_val, "MAE":mae, "MSE":mse}



def calculate_mri_image_di(
    npy_file: str,
    mat_file: str,
    mat_var_name: str = "image",
    auto_align: bool = True,  # 自动对齐开关（默认开启）
    show_plot: bool = True
):
    """
    终极版：自动像素对齐 + MRI仿真图像量化对比
    解决：图像偏左/偏上/偏移一格问题
    """
    # ===================== 1. 加载自定义 .npy 图像 =====================
    img_custom = np.load(npy_file)
    img_custom = np.squeeze(img_custom)
    img_custom = np.abs(img_custom).astype(np.float64)

    # ===================== 2. 加载 KomaMRI 图像 =====================
    with h5py.File(mat_file, "r") as f:
        print(f"✅ .mat 文件变量：{list(f.keys())}")
        struct_data = f[mat_var_name][:]
        img_koma = struct_data['real'] + 1j * struct_data['imag']
        img_koma = np.squeeze(img_koma.T)
        img_koma = np.abs(img_koma)
        img_koma = np.abs(img_koma[0:64,:])

    # ===================== 3. 核心：自动像素对齐（修复偏左一格） =====================
    if auto_align:
        # 计算两幅图像的平移偏移量 (行偏移, 列偏移)
        shift_vec, _, _ = phase_cross_correlation(img_koma, img_custom, upsample_factor=10)
        print(f"🔍 检测到图像偏移：向上{shift_vec[0]:.1f}格，向左{shift_vec[1]:.1f}格")
        
        # 平移校正你的图像（对齐到KomaMRI）
        img_custom_aligned = shift(img_custom, shift_vec, mode='constant', cval=0)
        print(f"✅ 图像已自动对齐完成！")
    else:
        img_custom_aligned = img_custom

    # ===================== 4. 归一化 =====================
    def normalize(img):
        img = img - np.min(img)
        return img / (np.max(img) + 1e-8)

    img1_norm = normalize(img_custom_aligned)
    img2_norm = normalize(img_koma)

    # ===================== 5. 量化指标 =====================
    flat1 = img1_norm.flatten()
    flat2 = img2_norm.flatten()

    mae = np.mean(np.abs(flat1 - flat2))
    mse = np.mean((flat1 - flat2) ** 2)
    psnr_val = psnr(img1_norm, img2_norm, data_range=1.0)
    ssim_val = ssim(img1_norm, img2_norm, data_range=1.0)
    mi_val = mutual_info_score((flat1 * 255).astype(int), (flat2 * 255).astype(int))

    # ===================== 6. 打印结果 =====================
    print("=" * 60)
    print("📊 对齐后 MRI 仿真图像量化对比结果")
    print("=" * 60)
    print(f"图像尺寸: {img1_norm.shape}")
    print(f"平均绝对误差 (MAE):    {mae:.6f}")
    print(f"均方误差 (MSE):        {mse:.6f}")
    print(f"峰值信噪比 (PSNR):     {psnr_val:.2f} dB")
    print(f"结构相似性 (SSIM):     {ssim_val:.6f}")
    print(f"互信息 (MI):           {mi_val:.4f}")
    print("=" * 60)

    # ===================== 7. 可视化：对齐前后对比 =====================
    if show_plot:
        plt.figure(figsize=(20, 5))
        
        # 原始偏移图像
        plt.subplot(1, 4, 1)
        plt.imshow(normalize(img_custom), cmap='gray')
        plt.title('自定义仿真\n(偏移前)')
        plt.axis('off')

        # 对齐后图像
        plt.subplot(1, 4, 2)
        plt.imshow(img1_norm, cmap='gray')
        plt.title('自定义仿真\n(自动对齐后)')
        plt.axis('off')

        # KomaMRI标准图
        plt.subplot(1, 4, 3)
        plt.imshow(img2_norm, cmap='gray')
        plt.title('KomaMRI')
        plt.axis('off')

        # 差异图
        plt.subplot(1, 4, 4)
        plt.imshow(np.abs(img1_norm - img2_norm), cmap='jet')
        plt.colorbar(label='差异值')
        plt.title('对齐后差异热力图')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return {
        "shape": img1_norm.shape, "MAE": mae, "MSE": mse,
        "PSNR": psnr_val, "SSIM": ssim_val, "MI": mi_val,
        "shift": shift_vec  # 偏移量
    }


def calculate_mri_image_diff(
    npy_file: str,
    mat_file: str,
    origin: str,
    mat_var_name: str = "image",  # 你的.mat变量名是 image！
    show_plot: bool = True
):
    """
    终极修复：适配 KomaMRI 结构体复数格式 + 消除所有报错/警告
    对比 自定义.npy 和 KomaMRI.mat 图像差异
    """
    # ===================== 1. 加载并处理自定义 .npy 图像 =====================
    img_custom = np.load(npy_file)
    img_custom = np.squeeze(img_custom)
    # 修复：先取绝对值，再转类型 → 消除 ComplexWarning
    img_custom = np.abs(img_custom).astype(np.float64)

    img_origin = np.load(origin)
    img_origin = np.squeeze(img_origin)
    img_origin = np.abs(img_origin).astype(np.float64)

    # ===================== 2. 【核心】读取 KomaMRI 结构体复数 =====================
    with h5py.File(mat_file, "r") as f:
        print(f"✅ .mat 文件变量：{list(f.keys())}")
        # 读取结构体数据 (real + imag)
        struct_data = f[mat_var_name][:]
        
        # 手动合成复数数组（解决结构体复数无法转换问题）
        img_koma = struct_data['real'] + 1j * struct_data['imag']
        # 维度修正 + 去冗余
        img_koma = np.squeeze(img_koma.T)
        # 取幅度
        img_koma = np.abs(img_koma)

    # ===================== 3. 尺寸校验 =====================
    if img_custom.shape != img_koma.shape:
        raise ValueError(f"尺寸不匹配！你的：{img_custom.shape}，KomaMRI：{img_koma.shape}")

    # ===================== 4. 归一化（消除 0-10 vs 0-30000 差异） =====================
    def normalize(img):
        img = img - np.min(img)
        return img / (np.max(img) + 1e-8)

    img1_norm = normalize(img_custom)
    img2_norm = normalize(img_koma)
    img3_norm = normalize(img_origin)
    # ===================== 5. 计算量化指标 =====================
    flat1 = img1_norm.flatten()
    flat2 = img2_norm.flatten()

    mae = np.mean(np.abs(flat1 - flat2))
    mse = np.mean((flat1 - flat2) ** 2)
    psnr_val = psnr(img1_norm, img2_norm, data_range=1.0)
    ssim_val = ssim(img1_norm, img2_norm, data_range=1.0)
    mi_val = mutual_info_score((flat1 * 255).astype(int), (flat2 * 255).astype(int))

    # ===================== 6. 打印结果 =====================
    print("=" * 60)
    print("📊 MRI 仿真图像量化对比结果")
    print("=" * 60)
    print(f"图像尺寸: {img1_norm.shape}")
    print(f"平均绝对误差 (MAE):    {mae:.6f}")
    print(f"均方误差 (MSE):        {mse:.6f}")
    print(f"峰值信噪比 (PSNR):     {psnr_val:.2f} dB")
    print(f"结构相似性 (SSIM):     {ssim_val:.6f}")
    print(f"互信息 (MI):           {mi_val:.4f}")
    print("=" * 60)
    print("✅ SSIM ≈ 1.0 说明两幅图像完全一致！")

    # ===================== 7. 可视化 =====================
    if show_plot:
        plt.figure(figsize=(18, 5), constrained_layout=True)
        plt.subplot(131), plt.imshow(img2_norm, cmap='gray'), plt.title('KomaMRI'), plt.axis('off')
        plt.subplot(132), plt.imshow(img1_norm, cmap='gray'), plt.title('Simulation'), plt.axis('off')
        plt.subplot(133), plt.imshow(img3_norm, cmap='gray'), plt.title('Simulation'), plt.axis('off')
        #plt.subplot(133), plt.imshow(np.abs(img1_norm-img2_norm), cmap='jet'), plt.title('Difference Heatmap'), plt.axis('off')
        #plt.colorbar(label='Difference Value', shrink=0.72)
        plt.show()

    return {
        "shape": img1_norm.shape, "MAE": mae, "MSE": mse,
        "PSNR": psnr_val, "SSIM": ssim_val, "MI": mi_val
    }


# ========== 请修改为你的文件路径 ==========
if __name__ == "__main__":
    result = calculate_mri_image_diff(
        npy_file="image_recon_gre.npy",       # 你的仿真结果路径
        mat_file="koma_gre.mat",    # KomaMRI结果路径
        mat_var_name="image",
        origin="image_recon_ideal.npy"               # ⚠️ 必须修改！.mat文件中的变量名
    )