# GRE Artifact Optimization Report

## 1. Scope

This report targets the severe vertical stripe and alias-like artifacts observed in the Cartesian GRE forward simulation.
The investigation followed the requested order:

1. inspect the project code and main execution path;
2. inspect the local PyPulseq implementation;
3. compare GRE and EPI sequence settings;
4. prioritize parameter-side fixes;
5. only then assess deeper simulation and reconstruction risks.

## 2. Code Path Review

### main.py

The previous `main.py` path used:

- `write_gre_label_sequence()` to generate the GRE sequence;
- `simulate()` to generate ADC samples;
- `reconstruct_image()` to run NUFFT reconstruction through `sigpy`.

For Cartesian GRE this reconstruction path is unnecessary and increases risk.
A direct FFT is the correct first-line reconstruction method.

### PyPulseq findings

The local installed library is `pypulseq 1.5.0.post1`.
The simulator correctly works on `Sequence.get_block()` results, where the relevant block fields are:

- `block_duration`
- `rf`
- `gx`, `gy`, `gz`
- `adc`
- `label`
- `soft_delay`

`Sequence.calculate_kspace()` confirmed that the GRE readout trajectory is a standard Cartesian line-by-line traversal with monotonic `kx` samples and constant `ky` per readout line.

### GRE vs EPI comparison

- EPI uses one excitation followed by many readouts, so it does not repeatedly rely on spoiled-GRE steady-state behaviour.
- GRE uses a new RF excitation every phase-encode line and therefore is much more sensitive to RF spoiling, TR transient behaviour, and the simulator's treatment of residual transverse coherences.
- The current GRE generators hard-coded classic RF spoiling (`117 deg`) before tuning.

## 3. Parameter-Side Experiments

All comparisons below used FFT reconstruction only, so the sequence behaviour could be judged without NUFFT confounds.

### 32x32 diagnostic metrics

- `baseline`: `nrmse=0.6040`, `col_err=0.4347`, `row_err=0.2718`
- `no_rf_spoil`: `nrmse=0.5267`, `col_err=0.3544`, `row_err=0.2582`
- `dummy8`: `nrmse=0.6101`, `col_err=0.4795`, `row_err=0.2510`
- `dummy8_no_spoil`: `nrmse=0.5411`, `col_err=0.3765`, `row_err=0.2639`
- `long_tr`: `nrmse=0.6657`, `col_err=0.5298`, `row_err=0.3676`
- `strong_spoil`: `nrmse=0.6184`, `col_err=0.4918`, `row_err=0.2538`

Additional no-RF-spoil tuning:

- `no_rf_spoil_flip5`: worse than `no_rf_spoil`
- `no_rf_spoil_short_ro`: worse than `no_rf_spoil`
- `no_rf_spoil_flip5_short_ro`: worse than `no_rf_spoil`

### Parameter conclusion

The single best parameter change was:

- disable RF spoiling for the current GRE demo path (`rf_spoiling_inc_deg = 0.0`)

Dummy scans, longer TR, and stronger spoilers did not solve the artifact and in several cases made the image worse.

## 4. Deeper Cause After Parameter Tuning

Parameter tuning improved the image substantially, but did not completely remove residual striping.
This points to a deeper modeling limitation:

- the current forward model uses a single isochromat per voxel;
- classic spoiled GRE assumes intravoxel dephasing and/or multiple sub-voxel isochromats;
- with only one isochromat per voxel, RF spoiling can create unrealistic coherent steady-state behaviour and strong line-to-line artifacts.

This explains why removing RF spoiling improves the image even though RF spoiling is physically reasonable on a scanner.

## 5. Reconstruction Review

### Previous risk

`reconstruct_image()` imported `sigpy` at module import time and used NUFFT for Cartesian data.
This is not the right primary reconstruction path for the current GRE demo.

### Current action

- `main.py` now uses `reconstruct_image_fft()` for the Cartesian GRE demo.
- `reconn.py` now imports `sigpy` lazily inside `reconstruct_image()` so FFT-only workflows do not pay the dependency cost.

## 6. Implemented Code Changes

### Sequence updates

Both GRE generators now accept:

- `rf_spoiling_inc_deg`
- `dummy_scans`

Files:

- `Sequence/write_gre.py`
- `Sequence/write_gre_label.py`

### Demo path updates

`main.py` now:

- disables RF spoiling for the demo GRE run;
- uses FFT reconstruction instead of NUFFT.

### Reconstruction update

`reconn.py` now lazily imports `sigpy` inside the NUFFT function.

## 7. Image-Level Result

A 64x64 comparison was generated after the code changes:

- baseline GRE still shows strong periodic vertical striping and horizontal banding;
- optimized GRE significantly suppresses the dominant vertical stripe artifact and restores the main object shape and the bright marker block much more clearly.

Residual faint striping remains, so the artifact is improved substantially but not fully eliminated.

## 8. Best-Practice Recommendations

1. Use FFT, not NUFFT, as the default reconstruction for Cartesian GRE and EPI.
2. Keep RF spoiling configurable instead of hard-coding it.
3. For spoiled GRE research, upgrade the phantom model to multiple isochromats or sub-voxel phase packets.
4. Validate GRE changes with both image-space inspection and k-space profile checks.
5. Keep sequence parameter tuning separate from reconstruction tuning during debugging.

## 9. Follow-Up Work

To remove the remaining residual artifact, the next priority should be:

1. add multi-isochromat / sub-voxel dephasing support for spoiled GRE;
2. then revisit whether RF spoiling can be re-enabled safely;
3. optionally add an automatic Cartesian-vs-NUFFT reconstruction dispatcher.
