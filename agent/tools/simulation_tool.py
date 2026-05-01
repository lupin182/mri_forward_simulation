
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import device_manager
device_manager.disable_cupy()

from agent.tools.base_tool import MRISimulationBaseTool
from simulate import SimulationConfig, simulate
from Sequence.write_gre import write_gre_sequence
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_se import write_se_sequence
from Sequence.write_epi import write_epi_sequence
from Sequence.write_epi_se import write_epi_se_sequence
from Sequence.write_epi_label import write_epi_label_sequence
from agent.tools.phantom_tool import get_cached_phantom
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_kspace_cache = None
_seq_cache = None
_kspace_figure_cache = None

class RunSimulationTool(MRISimulationBaseTool):
    name: str = "run_simulation"
    description: str = """运行MRI模拟（需要先生成体模）。
    参数说明：
    - sequence_type: 序列类型，可选 'gre'（梯度回波）、'gre_label'（带标签的梯度回波，默认）、'se'（自旋回波）、'epi'（平面回波）、'epi_se'（平面回波自旋回波）、'epi_label'（带标签的平面回波）
    - tr: 重复时间（秒），默认0.1
    - te: 回波时间（秒），默认0.02
    - fine_dt: 精细时间步长（秒），默认1e-5
    - return_figure: 是否返回matplotlib figure对象用于Streamlit展示，默认True
    """

    def _run(self, query: str) -> str:
        global _kspace_cache, _seq_cache, _kspace_figure_cache
        
        cached_data = get_cached_phantom()
        if cached_data is None:
            return json.dumps({"status": "error", "message": "请先生成体模"})
        
        phantom, _, _, _ = cached_data
        
        params = json.loads(query)
        sequence_type = params.get('sequence_type', 'gre_label')
        tr = params.get('tr', 0.1)
        te = params.get('te', 0.02)
        fine_dt = params.get('fine_dt', 1e-5)
        return_figure = params.get('return_figure', True)

        if sequence_type == 'gre':
            seq = write_gre_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'gre_label':
            seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'se':
            seq = write_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, te=te)
        elif sequence_type == 'epi':
            seq = write_epi_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        elif sequence_type == 'epi_se':
            seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), te=te)
        elif sequence_type == 'epi_label':
            seq = write_epi_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        else:
            return json.dumps({"status": "error", "message": f"未知的序列类型 '{sequence_type}'"})

        k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=fine_dt))
        _kspace_cache = k_space_signal
        _seq_cache = seq
        
        if return_figure and k_space_signal is not None:
            # 尝试将一维信号重塑为二维
            Nx, Ny = phantom.Nx, phantom.Ny
            try:
                # 先尝试直接重塑
                kspace_2d = np.abs(k_space_signal).reshape(Ny, Nx)
            except:
                try:
                    # 尝试截断或填充
                    total_len = Nx * Ny
                    if len(k_space_signal) >= total_len:
                        kspace_2d = np.abs(k_space_signal[:total_len]).reshape(Ny, Nx)
                    else:
                        kspace_2d = np.zeros((Ny, Nx), dtype=np.float64)
                        kspace_2d.flat[:len(k_space_signal)] = np.abs(k_space_signal)
                except:
                    # 如果都不行，显示一维信号
                    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                    ax.set_title("k-space Signal (1D)")
                    ax.plot(np.abs(k_space_signal))
                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel("Magnitude")
                    plt.tight_layout()
                    _kspace_figure_cache = fig
                    return
            
            # 显示二维k空间
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].set_title("k-space Magnitude")
            axes[0].imshow(kspace_2d, cmap='gray')
            axes[0].axis('off')
            axes[1].set_title("k-space Log Magnitude")
            axes[1].imshow(np.log1p(kspace_2d), cmap='gray')
            axes[1].axis('off')
            plt.tight_layout()
            _kspace_figure_cache = fig

        result = {
            "status": "success",
            "sequence_type": sequence_type,
            "phantom_shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "k_space_shape": k_space_signal.shape
        }

        return json.dumps(result, ensure_ascii=False)

def get_cached_kspace():
    global _kspace_cache
    return _kspace_cache

def get_cached_seq():
    global _seq_cache
    return _seq_cache

def get_cached_kspace_figure():
    global _kspace_figure_cache
    return _kspace_figure_cache

def clear_simulation_cache():
    global _kspace_cache, _seq_cache, _kspace_figure_cache
    _kspace_cache = None
    _seq_cache = None
    _kspace_figure_cache = None

