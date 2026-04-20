"""Export project phantoms to MRiLab-compatible ``VObj`` MAT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

DEFAULT_GYRO_RAD_PER_T = 2.0 * np.pi * 42.576e6


@dataclass(slots=True)
class MRiLabPhantomExport:
    """Summary of an exported MRiLab phantom."""

    output_path: Path
    type_num: int
    shape_yxz: tuple[int, int, int]
    x_dim_res_m: float
    y_dim_res_m: float
    z_dim_res_m: float


def _to_numpy(array: Any) -> np.ndarray:
    if array is None:
        raise ValueError("Expected a NumPy/CuPy-like array, got None.")
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


def _normalize_type_spin_layout(array: Any, name: str) -> np.ndarray:
    arr = _to_numpy(array)
    if arr.ndim == 3:
        arr = arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 4:
        arr = arr[:, np.newaxis, ...]
    elif arr.ndim != 5:
        raise ValueError(
            f"{name} must have shape (Nz, Nx, Ny), (TypeNum, Nz, Nx, Ny), "
            f"or (TypeNum, SpinNum, Nz, Nx, Ny); got {arr.shape}."
        )
    return arr.astype(np.float64, copy=False)


def _collapse_spin_packets(array: np.ndarray, name: str) -> np.ndarray:
    if array.shape[1] == 1:
        return array[:, 0]

    ref = array[:, :1]
    if np.allclose(array, ref, atol=1e-8, rtol=0.0, equal_nan=True):
        return ref[:, 0]

    raise ValueError(
        f"MRiLab VObj does not store a SpinNum dimension. {name} has "
        f"{array.shape[1]} non-identical spin packets, so this phantom "
        "cannot be exported losslessly."
    )


def _extract_typewise_constant(array: Any, type_num: int, name: str) -> np.ndarray:
    arr = _to_numpy(array)
    if arr.ndim == 0:
        return np.full(type_num, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == type_num:
        return arr.astype(np.float64, copy=False)

    arr = _collapse_spin_packets(_normalize_type_spin_layout(arr, name), name)
    values = np.zeros(type_num, dtype=np.float64)
    for type_idx in range(type_num):
        data = np.asarray(arr[type_idx], dtype=np.float64)
        active = np.abs(data) > 1e-10
        if not np.any(active):
            continue
        ref = float(data[active][0])
        if not np.allclose(data[active], ref, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"{name} varies spatially within type {type_idx + 1}. "
                "MRiLab stores this field only once per type."
            )
        values[type_idx] = ref
    return values


def _infer_object_type(type_num: int) -> str:
    if type_num == 1:
        return "Water"
    return f"PyPulseq Export ({type_num} pools)"


def _prepare_phantom_inputs(
    *,
    phantom: Any | None,
    rho: Any | None,
    t1: Any | None,
    t2: Any | None,
    fov_x: float | None,
    fov_y: float | None,
    slice_thickness: float | None,
    b0_t: float | None,
    chem_shift_hz: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, Any | None]:
    if phantom is not None:
        rho = getattr(phantom, "rho", rho)
        t1 = getattr(phantom, "t1", t1)
        t2 = getattr(phantom, "t2", t2)
        fov_x = getattr(phantom, "fov_x", fov_x)
        fov_y = getattr(phantom, "fov_y", fov_y)
        slice_thickness = getattr(phantom, "slice_thickness", slice_thickness)
        b0_t = getattr(phantom, "B0", b0_t)
        if chem_shift_hz is None and hasattr(phantom, "CS"):
            chem_shift_hz = getattr(phantom, "CS")

    if rho is None or t1 is None or t2 is None:
        raise ValueError("rho, t1 and t2 are required unless a Phantom object is provided.")
    if fov_x is None or fov_y is None or slice_thickness is None:
        raise ValueError("fov_x, fov_y and slice_thickness are required.")
    if b0_t is None:
        b0_t = 3.0

    rho_arr = _normalize_type_spin_layout(rho, "rho")
    t1_arr = _normalize_type_spin_layout(t1, "t1")
    t2_arr = _normalize_type_spin_layout(t2, "t2")
    if rho_arr.shape != t1_arr.shape or rho_arr.shape != t2_arr.shape:
        raise ValueError(
            "rho, t1 and t2 must share the same shape after normalization. "
            f"Got {rho_arr.shape}, {t1_arr.shape} and {t2_arr.shape}."
        )

    return rho_arr, t1_arr, t2_arr, float(fov_x), float(fov_y), float(slice_thickness), float(b0_t), chem_shift_hz


def export_phantom_to_mrilab_mat(
    output_path: str | Path,
    *,
    phantom: Any | None = None,
    rho: Any | None = None,
    t1: Any | None = None,
    t2: Any | None = None,
    t2_star: Any | None = None,
    fov_x: float | None = None,
    fov_y: float | None = None,
    slice_thickness: float | None = None,
    b0_t: float | None = None,
    chem_shift_hz: Any | None = None,
    chem_shift_hz_per_t: Any | None = None,
    gyro_rad_per_t: float = DEFAULT_GYRO_RAD_PER_T,
    object_type: str | None = None,
) -> MRiLabPhantomExport:
    """Write a project phantom as a MRiLab ``VObj`` MAT file.

    Parameters
    ----------
    output_path
        Destination ``.mat`` path.
    phantom
        Optional project ``Phantom`` instance.
    rho, t1, t2
        Phantom property arrays. Supported layouts are
        ``(Nz, Nx, Ny)``, ``(TypeNum, Nz, Nx, Ny)`` and
        ``(TypeNum, SpinNum, Nz, Nx, Ny)``.
    t2_star
        Optional ``T2Star`` array. If omitted, ``T2Star`` is exported as ``T2``.
    fov_x, fov_y, slice_thickness
        Spatial sampling information in meters.
    b0_t
        Main field strength in Tesla. Used only when converting chemical shift
        from Hz to the MRiLab ``Hz/T`` convention.
    chem_shift_hz, chem_shift_hz_per_t
        Optional type-wise chemical shift information. ``chem_shift_hz`` may be
        supplied either as a scalar/type vector or in project phantom layout.
    """
    rho_arr, t1_arr, t2_arr, fov_x, fov_y, slice_thickness, b0_t, inferred_cs_hz = _prepare_phantom_inputs(
        phantom=phantom,
        rho=rho,
        t1=t1,
        t2=t2,
        fov_x=fov_x,
        fov_y=fov_y,
        slice_thickness=slice_thickness,
        b0_t=b0_t,
        chem_shift_hz=chem_shift_hz,
    )
    if chem_shift_hz is None:
        chem_shift_hz = inferred_cs_hz
    if chem_shift_hz is not None and chem_shift_hz_per_t is not None:
        raise ValueError("Specify only one of chem_shift_hz and chem_shift_hz_per_t.")

    rho_arr = _collapse_spin_packets(rho_arr, "rho")
    t1_arr = _collapse_spin_packets(t1_arr, "t1")
    t2_arr = _collapse_spin_packets(t2_arr, "t2")
    t2_star_arr = t2_arr if t2_star is None else _collapse_spin_packets(_normalize_type_spin_layout(t2_star, "t2_star"), "t2_star")
    if t2_star_arr.shape != t2_arr.shape:
        raise ValueError(f"t2_star must match t2 after normalization; got {t2_star_arr.shape} vs {t2_arr.shape}.")

    type_num, nz, nx, ny = rho_arr.shape
    x_dim_res = fov_x / nx
    y_dim_res = fov_y / ny
    z_dim_res = slice_thickness

    if chem_shift_hz_per_t is not None:
        chem_shift = _extract_typewise_constant(chem_shift_hz_per_t, type_num, "chem_shift_hz_per_t")
    elif chem_shift_hz is not None:
        chem_shift_hz_value = _extract_typewise_constant(chem_shift_hz, type_num, "chem_shift_hz")
        if np.any(np.abs(chem_shift_hz_value) > 0) and abs(b0_t) < 1e-12:
            raise ValueError("b0_t must be non-zero when chem_shift_hz is provided.")
        chem_shift = chem_shift_hz_value / b0_t
    else:
        chem_shift = np.zeros(type_num, dtype=np.float64)

    def to_mrilab_layout(array: np.ndarray) -> np.ndarray:
        return np.transpose(array, (3, 2, 1, 0))

    vobj = {
        "Gyro": float(gyro_rad_per_t),
        "Type": object_type or _infer_object_type(type_num),
        "TypeNum": int(type_num),
        "XDim": int(nx),
        "XDimRes": float(x_dim_res),
        "YDim": int(ny),
        "YDimRes": float(y_dim_res),
        "ZDim": int(nz),
        "ZDimRes": float(z_dim_res),
        "Rho": to_mrilab_layout(rho_arr),
        "T1": to_mrilab_layout(t1_arr),
        "T2": to_mrilab_layout(t2_arr),
        "T2Star": to_mrilab_layout(t2_star_arr),
        "ChemShift": chem_shift,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, {"VObj": vobj}, do_compression=True, oned_as="row")

    return MRiLabPhantomExport(
        output_path=output_path,
        type_num=type_num,
        shape_yxz=(ny, nx, nz),
        x_dim_res_m=x_dim_res,
        y_dim_res_m=y_dim_res,
        z_dim_res_m=z_dim_res,
    )

def generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64):
    """
    生成一个单切片非对称体模，用于验证方向和坐标系,这里体模数据是离散的,且简化为每个体素单类型单spin_packet
    1. 中心有一个大圆/圆柱 (密度 1.0)
    2. 有一个小方块/圆点 (密度 2.0) -> 用于定方向
    3. 背景有微弱信号 (密度 0.1)

    返回:
    Rho: (type, spin_packet, Nz, Nx, Ny) 密度
    T1:  (type, spin_packet, Nz, Nx, Ny) T1值
    T2:  (type, spin_packet, Nz, Nx, Ny) T2值
    """
    
    # 1. 初始化 (背景密度 0.1, T1=2s, T2=0.5s)
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.005


    # 2. 生成坐标网格 (注意这里为了生成数据，我们严格按照 Nz, Nx, Ny 顺序)
    # 这里的 x_idx 对应 Nx 维度，y_idx 对应 Ny 维度
    _, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx) - Nx/2,  # 居中坐标: -32 到 +32
        np.arange(Ny) - Ny/2, 
        indexing='ij'
    )

    # 3. 主体：中心大圆 (半径 16)
    # x^2 + y^2 < r^2
    mask_main = (x**2 + y**2) <= 16**2
    
    # 4. 标记物：右上角小点 (卫星)
    # 放在 X 正半轴, Y 正半轴 (例如 x=15, y=15 处)
    # 这是一个 6x6 的小方块
    mask_marker = (x > 10) & (x < 20) & (y > 10) & (y < 20)

    # 5. 赋值
    # 主体：水 (Rho=1.0, T1=1.0, T2=0.1)
    rho[mask_main] = 1.0
    t1[mask_main]  = 1.0
    t2[mask_main]  = 0.1

    # 标记物：高亮油点 (Rho=2.0, T1=0.5, T2=0.05 - 短T1更亮(在短TR下), 短T2)
    rho[mask_marker] = 2.0 
    t1[mask_marker]  = 0.5 
    t2[mask_marker]  = 0.05

    rho[0,10:15,50:55] = 2.0
    t1[0,10:15,50:55] = 0.5
    t2[0,10:15,50:55] = 0.05

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2
if __name__ == "__main__":
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=10, Nx=64, Ny=64)
    export_phantom_to_mrilab_mat(output_path="E:\毕业课题\mri_codex/test.mat", rho=rho, 
                                t1=t1, t2=t2, fov_x=0.256, fov_y=0.256, 
                                slice_thickness=0.005)
