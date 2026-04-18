
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import device_manager
device_manager.disable_cupy()

from agent.tools.base_tool import MRISimulationBaseTool
from phantom.make_phantom import (
    generate_simple_asymmetric_phantom,
    Phantom
)
from simulate import SimulationConfig, simulate
from recon import reconstruct_3d_cartesian_fft, plot_color_overlay
from Sequence.write_gre_label import write_gre_label_sequence
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

class ReconstructAndVisualizeTool(MRISimulationBaseTool):
    name: str = "reconstruct_and_visualize"
    description: str = """运行完整的MRI模拟流程并重建可视化图像。
    参数说明：
    - sequence_type: 序列类型，可选 'gre'（梯度回波）、'gre_label'（带标签的梯度回波，默认）、'se'（自旋回波）、'epi'（平面回波）、'epi_se'（平面回波自旋回波）、'epi_label'（带标签的平面回波）
    - phantom_type: 体模类型，可选 'asymmetric'（非对称，默认）、'ring'（圆环）、'sphere'（球体）
    - Nz: z轴切片数，默认1
    - Nx: x轴分辨率，默认64
    - Ny: y轴分辨率，默认64
    - fov_x: x轴视场（米），默认0.256
    - fov_y: y轴视场（米），默认0.256
    - slice_thickness: 切片厚度（米），默认0.004
    - tr: 重复时间（秒），默认0.1
    - te: 回波时间（秒），默认0.02
    - fine_dt: 精细时间步长（秒），默认1e-5
    - output_path: 保存图像的路径，默认不保存
    - show_plot: 是否显示图像，默认true
    """

    def _run(self, query: str) -> str:
        params = json.loads(query)
        
        phantom_type = params.get('phantom_type', 'asymmetric')
        Nz = params.get('Nz', 1)
        Nx = params.get('Nx', 64)
        Ny = params.get('Ny', 64)
        fov_x = params.get('fov_x', 0.256)
        fov_y = params.get('fov_y', 0.256)
        slice_thickness = params.get('slice_thickness', 0.004)
        sequence_type = params.get('sequence_type', 'gre_label')
        tr = params.get('tr', 0.1)
        te = params.get('te', 0.02)
        fine_dt = params.get('fine_dt', 1e-5)
        output_path = params.get('output_path', None)
        show_plot = params.get('show_plot', False)
        
        if output_path is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'mri_result.png')

        if phantom_type == 'asymmetric':
            rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=Nz, Nx=Nx, Ny=Ny)
        elif phantom_type == 'ring':
            from phantom.make_phantom import generate_simple_ring_phantom
            inner_radius = params.get('inner_radius', 10)
            outer_radius = params.get('outer_radius', 20)
            rho, t1, t2 = generate_simple_ring_phantom(Nz=Nz, Nx=Nx, Ny=Ny, inner_radius=inner_radius, outer_radius=outer_radius)
        elif phantom_type == 'sphere':
            from phantom.make_phantom import generate_simple_sphere_phantom
            radius = params.get('radius', 16)
            rho, t1, t2 = generate_simple_sphere_phantom(Nz=Nz, Nx=Nx, Ny=Ny, radius=radius)
        else:
            return f"Error: 未知的体模类型 '{phantom_type}'"

        phantom = Phantom(rho, t1, t2, fov_x=fov_x, fov_y=fov_y, slice_thickness=slice_thickness)

        if sequence_type == 'gre':
            from Sequence.write_gre import write_gre_sequence
            seq = write_gre_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'gre_label':
            seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'se':
            from Sequence.write_se import write_se_sequence
            seq = write_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, te=te)
        elif sequence_type == 'epi':
            from Sequence.write_epi import write_epi_sequence
            seq = write_epi_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        elif sequence_type == 'epi_se':
            from Sequence.write_epi_se import write_epi_se_sequence
            seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, te=te)
        elif sequence_type == 'epi_label':
            from Sequence.write_epi_label import write_epi_label_sequence
            seq = write_epi_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        else:
            return f"Error: 未知的序列类型 '{sequence_type}'"

        k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=fine_dt))
        k_traj_adc, _, _, _, _ = seq.calculate_kspace()
        image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)

        if show_plot or output_path:
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.title("Reconstruction")
            plt.imshow(np.abs(image_recon[0]), cmap='gray')
            plt.axis('off')

            plt.subplot(122)
            plt.title("Original Phantom")
            plt.imshow(rho[0, 0, 0], cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

        result = {
            "status": "success",
            "phantom_type": phantom_type,
            "sequence_type": sequence_type,
            "phantom_shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "output_path": output_path
        }

        return json.dumps(result)

