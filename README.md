# MRI Forward Simulation

本项目是一个磁共振成像前向模拟系统，支持体模生成与数据库管理、PyPulseq 序列生成、Bloch 方程前向模拟、k-space 采样、图像重建、RF/B0 伪影模拟，以及基于自然语言的智能代理工作台。

根目录只保留一个 Python 用户入口：[main.py](E:/毕业课题/mri_codex/main.py)。

## 功能概览

- 体模：支持 `asymmetric`、`sphere`、`ring` 三类内置体模，也支持本地体模数据库。
- 体模数据库：支持创建体模条目、导入合法体模数据文件、删除单个数据文件或整个体模目录。
- 序列：支持 `gre`、`gre_label`、`se`、`epi`、`epi_se`、`epi_label`。
- 前向模拟：基于 PyPulseq 序列事件和 Bloch 方程生成 k-space 信号。
- 图像重建：支持笛卡尔 FFT 重建和多通道 SOS 合成。
- 伪影：支持 RF 干扰伪影和 B0 不均匀场。
- 智能代理：支持命令行 ReAct agent 和 Streamlit 图形界面。
- 设备：CuPy/GPU 可用时自动使用 GPU，否则回退到 NumPy/CPU。

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CuPy 是可选依赖。未安装 CuPy 或没有可用 GPU 时，项目会自动使用 NumPy/CPU。

## Agent 配置

复制配置模板：

```powershell
Copy-Item .env.example .env
```

然后在根目录 `.env` 中填写：

```text
MRI_AGENT_API_KEY=your_api_key_here
MRI_AGENT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MRI_AGENT_MODEL=qwen3.5-plus
MRI_AGENT_TEMPERATURE=0.7
MRI_AGENT_MAX_TOKENS=2048
```

`.env` 已被 `.gitignore` 忽略，不会提交到 Git。

## 运行模拟

默认运行完整前向模拟：

```powershell
python main.py
```

显式运行模拟：

```powershell
python main.py simulate --sequence gre_label --phantom asymmetric --nx 64 --ny 64
```

使用数据库体模：

```powershell
python main.py simulate --phantom database --phantom-name test --sequence gre_label
```

常用参数示例：

```powershell
python main.py simulate `
  --phantom sphere `
  --sequence gre_label `
  --nx 64 `
  --ny 64 `
  --nz 1 `
  --fov-x 0.256 `
  --fov-y 0.256 `
  --slice-thickness 0.004 `
  --tr 0.1 `
  --te 0.02 `
  --fine-dt 1e-5 `
  --output-dir output/run_sphere_gre
```

输出文件包括 `kspace.npy`、`reconstruction.npy`、`reconstruction_magnitude.npy`、`summary.json` 和可选 PNG。

## 体模数据库位置

数据库根目录：

```text
mri_sim/phantom_depository/
```

每个体模是一个子目录：

```text
mri_sim/phantom_depository/<phantom_name>/
```

数据库索引文件：

```text
mri_sim/phantom_depository/info.txt
```

索引格式为每行一个体模：

```text
name:description
```

## 数据库管理命令

数据库管理命令放在 `simulate database` 下，不影响原有模拟命令。

列出数据库体模：

```powershell
python main.py simulate database list
```

创建体模条目：

```powershell
python main.py simulate database create --name brain_demo --description "Brain phantom with coil maps"
```

这个命令会创建：

```text
mri_sim/phantom_depository/brain_demo/
```

并向 `mri_sim/phantom_depository/info.txt` 追加或更新：

```text
brain_demo:Brain phantom with coil maps
```

导入一个合法体模数据文件：

```powershell
python main.py simulate database load --name brain_demo --data rho --file-path E:\data\rho.npy
```

导入后文件会被复制为规范文件名：

```text
mri_sim/phantom_depository/brain_demo/rho.npy
```

导入元数据：

```powershell
python main.py simulate database load --name brain_demo --data info --file-path E:\data\info.txt
```

导入线圈灵敏度图：

```powershell
python main.py simulate database load --name brain_demo --data rxCoilmg --file-path E:\data\rxCoilmg.npy
python main.py simulate database load --name brain_demo --data rxCoilpe --file-path E:\data\rxCoilpe.npy
```

删除指定体模中的某一个数据文件：

```powershell
python main.py simulate database delete --name brain_demo --data rxCoilpe
```

删除整个体模目录，并从数据库索引中移除该体模：

```powershell
python main.py simulate database delete --name brain_demo --all
```

## 合法数据名和文件名

`load` 和按数据删除的 `delete --data` 只接受以下合法数据名：

```text
rho
t1
t2
dB0
CS
dWRnd
txCoilmg
txCoilpe
rxCoilmg
rxCoilpe
info
```

对应保存文件名：

```text
rho       -> rho.npy
t1        -> t1.npy
t2        -> t2.npy
dB0       -> dB0.npy
CS        -> CS.npy
dWRnd     -> dWRnd.npy
txCoilmg  -> txCoilmg.npy
txCoilpe  -> txCoilpe.npy
rxCoilmg  -> rxCoilmg.npy
rxCoilpe  -> rxCoilpe.npy
info      -> info.txt
```

除 `info` 外，其它数据必须是可被 `numpy.load()` 读取的 `.npy` 文件。导入时会先验证文件可读，再复制到体模目录。

## 完整体模目录格式

```text
mri_sim/phantom_depository/full_example/
├── info.txt
├── rho.npy
├── t1.npy
├── t2.npy
├── dB0.npy
├── CS.npy
├── dWRnd.npy
├── txCoilmg.npy
├── txCoilpe.npy
├── rxCoilmg.npy
└── rxCoilpe.npy
```

必需文件：

- `rho.npy`：质子密度。
- `t1.npy`：T1 弛豫时间，单位秒。
- `t2.npy`：T2 弛豫时间，单位秒。

可选文件：

- `dB0.npy`：B0 不均匀场。
- `CS.npy`：化学位移。
- `dWRnd.npy`：T2* 随机离共振项。
- `txCoilmg.npy`：发射线圈幅度灵敏度图。
- `txCoilpe.npy`：发射线圈相位灵敏度图，单位弧度。
- `rxCoilmg.npy`：接收线圈幅度灵敏度图。
- `rxCoilpe.npy`：接收线圈相位灵敏度图，单位弧度。

基础体模数组支持：

```text
(Nz, Nx, Ny)
(TypeNum, SpinNum, Nz, Nx, Ny)
```

`dB0.npy`、`CS.npy`、`dWRnd.npy` 必须与最终体模数组形状一致：

```text
(TypeNum, SpinNum, Nz, Nx, Ny)
```

如果基础体模使用 `(Nz, Nx, Ny)`，这些可选数组建议保存为 `(1, 1, Nz, Nx, Ny)`。

线圈灵敏度图形状：

```text
txCoilmg.npy: (TxCoilNum, Nz, Nx, Ny)
txCoilpe.npy: (TxCoilNum, Nz, Nx, Ny)
rxCoilmg.npy: (RxCoilNum, Nz, Nx, Ny)
rxCoilpe.npy: (RxCoilNum, Nz, Nx, Ny)
```

## 体模 info.txt

体模目录中的 `info.txt` 用于保存模拟元数据，格式为 `key: value`：

```text
description: Full database phantom example
fov_x: 0.256
fov_y: 0.256
slice_thickness: 0.004
B0: 3.0
RxCoilNum: 8
TxCoilNum: 1
```

有效字段：

- `description`：体模描述。
- `fov_x`：x 方向 FOV，单位米，默认 `0.256`。
- `fov_y`：y 方向 FOV，单位米，默认 `0.256`。
- `slice_thickness`：层厚，单位米，默认 `0.004`。
- `B0`：主磁场强度，单位 T，默认 `3.0`。
- `RxCoilNum`：接收线圈数；未写时从接收线圈图第一维推断。
- `TxCoilNum`：发射线圈数；未写时从发射线圈图第一维推断。

根 CLI 对数据库体模的 FOV 处理规则：

- 命令行传入 `--fov-x`、`--fov-y`、`--slice-thickness` 时，命令行参数优先。
- 否则使用体模目录中的 `info.txt`。
- 如果二者都没有，则使用默认值。

## 完整数据库示例

下面脚本创建一个覆盖全部合法数据范围的 `full_example`：

```python
from pathlib import Path

import numpy as np

root = Path("mri_sim/phantom_depository/full_example")
root.mkdir(parents=True, exist_ok=True)

Nz, Nx, Ny = 1, 32, 32
TxCoilNum = 1
RxCoilNum = 4

y, x = np.ogrid[:Nx, :Ny]
mask = ((x - Ny / 2) ** 2 + (y - Nx / 2) ** 2) <= 10 ** 2

rho = np.zeros((Nz, Nx, Ny), dtype=np.float32)
t1 = np.ones((Nz, Nx, Ny), dtype=np.float32) * 1.0
t2 = np.ones((Nz, Nx, Ny), dtype=np.float32) * 0.08
rho[0, mask] = 1.0

np.save(root / "rho.npy", rho)
np.save(root / "t1.npy", t1)
np.save(root / "t2.npy", t2)

expanded_shape = (1, 1, Nz, Nx, Ny)
np.save(root / "dB0.npy", np.zeros(expanded_shape, dtype=np.float32))
np.save(root / "CS.npy", np.zeros(expanded_shape, dtype=np.float32))
np.save(root / "dWRnd.npy", np.zeros(expanded_shape, dtype=np.float32))

tx_mag = np.ones((TxCoilNum, Nz, Nx, Ny), dtype=np.float32)
tx_phase = np.zeros((TxCoilNum, Nz, Nx, Ny), dtype=np.float32)
rx_mag = np.zeros((RxCoilNum, Nz, Nx, Ny), dtype=np.float32)
rx_phase = np.zeros((RxCoilNum, Nz, Nx, Ny), dtype=np.float32)

for coil in range(RxCoilNum):
    angle = 2 * np.pi * coil / RxCoilNum
    rx_mag[coil, 0] = 1.0 + 0.25 * np.cos(angle) * (x - Ny / 2) / Ny
    rx_phase[coil, 0] = angle

np.save(root / "txCoilmg.npy", tx_mag)
np.save(root / "txCoilpe.npy", tx_phase)
np.save(root / "rxCoilmg.npy", rx_mag)
np.save(root / "rxCoilpe.npy", rx_phase)

(root / "info.txt").write_text(
    "\n".join(
        [
            "description: Full database phantom example",
            "fov_x: 0.256",
            "fov_y: 0.256",
            "slice_thickness: 0.004",
            "B0: 3.0",
            f"RxCoilNum: {RxCoilNum}",
            f"TxCoilNum: {TxCoilNum}",
        ]
    ),
    encoding="utf-8",
)
```

运行：

```powershell
python main.py simulate --phantom database --phantom-name full_example --sequence gre_label --no-plot
```

## Agent 数据库工具

启动命令行 agent：

```powershell
python main.py agent-cli
```

Agent 只负责列出和加载已有数据库体模，不提供创建、导入文件或删除数据库体模的管理能力。数据库管理请使用 `python main.py simulate database ...`。

列出：

```json
{"tool": "list_phantom_database", "params": {}}
```

加载体模用于模拟：

```json
{"tool": "load_phantom_from_database", "params": {"phantom_name": "brain_demo"}}
```

加载体模后继续调用模拟和重建：

```json
{"tool": "run_simulation", "params": {"sequence_type": "gre_label", "fine_dt": 1e-5}}
```

```json
{"tool": "reconstruct_image", "params": {"output_path": "output/mri_result.png"}}
```

## 伪影模拟

RF 伪影：

```powershell
python main.py simulate `
  --rf-artifact `
  --rf-noise-freq 127700000 `
  --rf-noise-amp 5.0 `
  --bg-noise-amp 1.0
```

B0 不均匀场：

```powershell
python main.py simulate `
  --b0-artifact `
  --b0-mode linear `
  --b0-delta-ppm 0.5 `
  --b0-axis x
```

如果数据库体模已经包含 `dB0.npy`，默认会使用数据库中的 B0 不均匀场；只有显式传入 `--b0-artifact` 时，才会在加载后重新生成并覆盖当前 `phantom.dB0`。

## Streamlit UI

```powershell
python main.py agent-ui
```

指定端口：

```powershell
python main.py agent-ui --server-port 8501
```

## Python API

```python
from mri_sim.phantom_database import load_phantom_dataset

dataset = load_phantom_dataset("test")
phantom = dataset.build_phantom()
```

## 验证

静态解析检查：

```powershell
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]; print('ok')"
```

小矩阵 smoke test：

```powershell
python main.py simulate --sequence gre_label --phantom asymmetric --nx 8 --ny 8 --fine-dt 1e-5 --no-plot
```

数据库 smoke test：

```powershell
python main.py simulate --phantom database --phantom-name test --sequence gre_label --fine-dt 1e-5 --no-plot
```

## 注意事项

- `output/`、`.env`、缓存目录和编辑器配置已被 `.gitignore` 忽略。
- 删除整个体模目录会同时从 `mri_sim/phantom_depository/info.txt` 移除该体模索引。
- 所有用户入口统一通过根目录 `main.py` 调用。
