
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

_kspace_cache = None
_seq_cache = None

class RunSimulationTool(MRISimulationBaseTool):
    name: str = "run_simulation"
    description: str = """运行MRI模拟（需要先生成体模）。
    参数说明：
    - sequence_type: 序列类型，可选 'gre'（梯度回波）、'gre_label'（带标签的梯度回波，默认）、'se'（自旋回波）、'epi'（平面回波）、'epi_se'（平面回波自旋回波）、'epi_label'（带标签的平面回波）
    - tr: 重复时间（秒），默认0.1
    - te: 回波时间（秒），默认0.02
    - fine_dt: 精细时间步长（秒），默认1e-5
    """

    def _run(self, query: str) -> str:
        global _kspace_cache, _seq_cache
        
        cached_data = get_cached_phantom()
        if cached_data is None:
            return json.dumps({"status": "error", "message": "请先生成体模"})
        
        phantom, _, _, _ = cached_data
        
        params = json.loads(query)
        sequence_type = params.get('sequence_type', 'gre_label')
        tr = params.get('tr', 0.1)
        te = params.get('te', 0.02)
        fine_dt = params.get('fine_dt', 1e-5)

        if sequence_type == 'gre':
            seq = write_gre_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'gre_label':
            seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, tr=tr, te=te)
        elif sequence_type == 'se':
            seq = write_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, te=te)
        elif sequence_type == 'epi':
            seq = write_epi_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        elif sequence_type == 'epi_se':
            seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, te=te)
        elif sequence_type == 'epi_label':
            seq = write_epi_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx, fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz)
        else:
            return json.dumps({"status": "error", "message": f"未知的序列类型 '{sequence_type}'"})

        k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=fine_dt))
        _kspace_cache = k_space_signal
        _seq_cache = seq

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

def clear_simulation_cache():
    global _kspace_cache, _seq_cache
    _kspace_cache = None
    _seq_cache = None

