# Bloch Kernel Validation Report

Generated at: 2026-04-11 21:11:47

## Code-reading conclusions

- `contrast_bloch_kernel/bloch_utils.py::solve_bloch_implicit()` is an implicit finite-difference Bloch update, not a `solve_ivp`-style ODE integrator.
- The reference solver and `bloch_kernel.py` both operate in a rotating-frame formulation.
- For a fair apples-to-apples comparison, the reference outputs were aligned to the candidate convention with:
  - `My_candidate_frame = -My_reference`
  - `rf_hz = -gamma_hz_per_t * B1_t / (2*pi)` when driving `BlochKernel()`
  - `dWRnd = dOmega_khz * 1e3` to match the reference off-resonance numerics

## Validation setup

- Model: single voxel, single spin, no gradients, no B0 inhomogeneity, one transmit coil with unit sensitivity
- RF pulse: 1.000 ms sinc, 5 lobes, peak 355.000 uT
- Relaxation: T1 = 250.0 ms, T2 = 70.0 ms
- Off-resonance for FID: 1.500 kHz in the reference convention
- Time step: 0.0010 ms
- Excitation comparison figure: `docs/bloch_kernel_excitation_comparison.png`
- FID comparison figure: `docs/bloch_kernel_fid_comparison.png`

## Excitation-stage metrics

| Signal | RMSE | MAE | Corr | Cosine | Max | Peak time [ms] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mx | 0.000960 | 0.000728 | 0.999998 | 0.999999 | 0.001494 | 0.566000 |
| My | 0.003355 | 0.001875 | 0.999971 | 0.999986 | 0.012370 | 0.479000 |
| Mz | 0.003248 | 0.001538 | 0.999978 | 0.999990 | 0.012187 | 0.522000 |

Overall excitation vector metrics:

- RMSE: 0.002752
- MAE: 0.001381
- Correlation: 0.999986
- Cosine similarity: 0.999989
- Maximum absolute error: 0.012370 at 0.479000 ms

Pulse-end magnetization at `t = 1.000` ms:

- Reference: `[-0.698208, 0.705053, 0.039738]`
- Candidate: `[-0.699283, 0.703962, 0.039652]`
- Difference: `[-0.001075, -0.001092, -0.000086]`

## FID-stage metrics

Complex FID signal metrics were computed on the aligned `Mxy = Mx + i My` trace, with correlation and cosine similarity evaluated on the concatenated real and imaginary parts.

- Complex RMSE: 0.001932
- Complex MAE: 0.001922
- Complex correlation: 0.999998
- Complex cosine similarity: 0.999998
- Complex maximum absolute error: 0.002221 at 20.000000 ms after pulse end

FID magnitude metrics:

- Magnitude RMSE: 0.000022
- Magnitude MAE: 0.000022
- Magnitude correlation: 1.000000
- Magnitude cosine similarity: 1.000000
- Magnitude maximum absolute error: 0.000024 at 20.000000 ms after pulse end

## Assessment

The new `bloch_kernel.py` implementation is consistent with the reference solver for this single-voxel 90-degree excitation + FID experiment after aligning the sign/unit conventions identified in the code review. The excitation-stage component RMSE values stay at the 1e-3 level, and the complex FID agreement is also at the 1e-3 level with correlation and cosine similarity essentially equal to 1.

The largest residual differences concentrate near the rapid mid-pulse rotation interval rather than the free-decay tail. This is expected because the candidate kernel uses exact short-step rotations with split relaxation, while the reference uses an implicit finite-difference update.
