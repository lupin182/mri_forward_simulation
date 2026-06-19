
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mri_sim.device_manager import disable_cupy
disable_cupy()

from agent.tools.base_tool import MRISimulationBaseTool
from mri_sim.reconstruction import reconstruct_3d_cartesian_fft
from agent.tools.phantom_tool import get_cached_phantom
from agent.tools.simulation_tool import get_cached_kspace, get_cached_seq
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_image_cache = None
_figure_cache = None

class ReconstructImageTool(MRISimulationBaseTool):
    name: str = "reconstruct_image"
    description: str = """重建MRI图像并保存（需要先运行模拟）。
    参数说明：
    - output_path: 保存图像的路径，默认保存到output目录
    - return_figure: 是否返回matplotlib figure对象用于Streamlit展示，默认True
    """

    def _run(self, query: str) -> str:
        global _image_cache, _figure_cache
        
        cached_phantom = get_cached_phantom()
        if cached_phantom is None:
            return json.dumps({"status": "error", "message": "请先生成体模"})
        
        phantom, rho, _, _ = cached_phantom
        
        k_space_signal = get_cached_kspace()
        seq = get_cached_seq()
        
        if k_space_signal is None or seq is None:
            return json.dumps({"status": "error", "message": "请先运行模拟"})
        
        params = json.loads(query)
        output_path = params.get('output_path', None)
        return_figure = params.get('return_figure', True)
        
        if output_path is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'mri_result.png')

        k_traj_adc, _, _, _, _ = seq.calculate_kspace()
        image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)
        _image_cache = image_recon

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].set_title("Reconstruction")
        axes[0].imshow(np.abs(image_recon[0]), cmap='gray')
        axes[0].axis('off')

        axes[1].set_title("Original Phantom")
        axes[1].imshow(rho[0, 0, 0], cmap='gray')
        axes[1].axis('off')

        plt.tight_layout()
        
        if return_figure:
            _figure_cache = fig
        else:
            plt.savefig(output_path)
            plt.close()

        result = {
            "status": "success",
            "phantom_shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "output_path": output_path
        }

        return json.dumps(result, ensure_ascii=False)

def get_cached_image():
    global _image_cache
    return _image_cache

def get_cached_figure():
    global _figure_cache
    return _figure_cache

def clear_recon_cache():
    global _image_cache, _figure_cache
    _image_cache = None
    _figure_cache = None

