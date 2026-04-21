import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==========================================
# 1. 读取 .mat 文件
# ==========================================
# 注意：请将 'your_file.mat' 替换为你的实际文件名
# 请使用 print(mat_data.keys()) 查看并替换 'data_variable_name' 为mat文件中实际的变量名
file_path = r"C:\Users\emd\AppData\Local\Temp\data_phantom.mat" 

import numpy as np
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. 使用 h5py 读取 v7.3 格式的 .mat 文件
# ==========================================

with h5py.File(file_path, 'r') as f:
    # 打印出文件内所有的变量名，方便你找到存数据的具体变量
    print("MAT文件中的变量名有:", list(f.keys()))
    
    # 请将 'data_variable_name' 替换为上面打印出来的实际变量名
    # 如果只有1个主要的变量，直接填入即可
    data_raw = np.array(f['phantom']['data'])

print(f"刚读取进来的数据 shape: {data_raw.shape}")

# MATLAB v7.3 读取进 Python 往往是转置的。
# 我们需要的是 N*8 的格式，如果读成了 8*N，就将其转置回来
if data_raw.shape[0] == 8 and data_raw.shape[1] > 8:
    data = data_raw.T
    print(f"转置后的数据 shape: {data.shape}")
else:
    data = data_raw
# ==========================================
# 2. 解析 8 列数据
# ==========================================
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
rho = data[:, 3]
T1 = data[:, 4]
T2 = data[:, 5]
T2s = data[:, 6]
delta_omega = data[:, 7]
print(np.sort(x))
# ==========================================
# 3. 计算 FOV 和 图像尺寸
# ==========================================
# FOV = 坐标最大值 - 坐标最小值 (单位：米)
fov_x = np.ptp(x)  # np.ptp 等价于 np.max() - np.min()
fov_y = np.ptp(y)
fov_z = np.ptp(z)

def get_spacing_and_dim(coords, fov):
    """
    通过寻找坐标数组中相邻元素的最小差值，来推算网格分辨率和矩阵维度。
    """
    unique_coords = np.unique(coords)
    if len(unique_coords) > 1:
        # 最小的正差值即为体素间距 (dx, dy 或 dz)
        spacing = np.min(np.diff(unique_coords))
        # 维度 = FOV / 体素间距 + 1
        dim = int(np.round(fov / spacing)) + 1
        return spacing, dim
    else:
        return 1.0, 1 # 处理单一平面(如2D仿真中z只有1个值)的情况

dx, nx = get_spacing_and_dim(x, fov_x)
dy, ny = get_spacing_and_dim(y, fov_y)
dz, nz = get_spacing_and_dim(z, fov_z)

print("-" * 30)
print(f"FOV (x, y, z): ({fov_x:.4f} m, {fov_y:.4f} m, {fov_z:.4f} m)")
print(f"网格分辨率 (dx, dy, dz): ({dx:.4f} m, {dy:.4f} m, {dz:.4f} m)")
print(f"图像尺寸 Matrix Size (nx, ny, nz): ({nx}, {ny}, {nz})")
print("-" * 30)

# ==========================================
# 4. 创建空图像并映射数据
# ==========================================
# 初始化全0的网格，原本文件中因为0被压缩掉的背景，现在会自动保持为0
img_rho = np.zeros((nx, ny, nz))
img_T1 = np.zeros((nx, ny, nz))
img_T2 = np.zeros((nx, ny, nz))
img_T2s = np.zeros((nx, ny, nz))
img_dw = np.zeros((nx, ny, nz))

# 将物理坐标映射为矩阵的整数索引 [0, N-1]
ix = np.round((x - np.min(x)) / dx).astype(int) if nx > 1 else np.zeros_like(x, dtype=int)
iy = np.round((y - np.min(y)) / dy).astype(int) if ny > 1 else np.zeros_like(y, dtype=int)
iz = np.round((z - np.min(z)) / dz).astype(int) if nz > 1 else np.zeros_like(z, dtype=int)

# 将值填入矩阵对应的位置
img_rho[ix, iy, iz] = rho
img_T1[ix, iy, iz] = T1
img_T2[ix, iy, iz] = T2
img_T2s[ix, iy, iz] = T2s
img_dw[ix, iy, iz] = delta_omega

# ==========================================
# 5. 可视化图像
# ==========================================
# 如果是3D数据，取Z轴的中心切片；如果是2D数据，nz=1，中心切片就是0
mid_z = nz // 2

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
titles = ['Rho (Proton Density)', 'T1', 'T2', 'T2*', 'Delta Omega']
images = [img_rho, img_T1, img_T2, img_T2s, img_dw]
cmaps = ['gray', 'magma', 'viridis', 'viridis', 'coolwarm']
np.save('img_rho.npy', img_rho)
np.save('img_T1.npy', img_T1)
np.save('img_T2.npy', img_T2)
np.save('img_T2s.npy', img_T2s)
np.save('img_dw.npy', img_dw)
for i, (ax, img, title, cmap) in enumerate(zip(axes, images, titles, cmaps)):
    # 提取切片。由于矩阵索引 [ix, iy] 中 x 对应行、y 对应列，我们用 .T 转置将其变为常规的图像坐标系
    slice_img = img[:, :, mid_z].T 
    
    # 绘制图像，origin='lower' 确保 y 轴的正方向朝上
    im = ax.imshow(slice_img, cmap=cmap, origin='lower')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    # 添加 Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()
