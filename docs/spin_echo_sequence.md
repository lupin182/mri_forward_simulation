# Spin Echo Sequence

## Overview

This repository now includes a standard Cartesian spin-echo sequence
implementation in `Sequence/write_se.py`.
The sequence is built with the local `pypulseq 1.5.0.post1` installation and
matches the project sequence style already used by the GRE and EPI generators.

The implemented timing follows the same physical structure as
`Sequence/se1.py`, which is the existing non-PyPulseq reference sequence:

1. 90 degree slice-selective excitation.
2. Readout, phase-encoding and slice-rephasing prephasers.
3. Delay to place the 180 degree pulse center at `TE / 2`.
4. 180 degree slice-selective refocusing pulse.
5. Delay to place the readout center at `TE`.
6. Cartesian readout with ADC.
7. Delay to complete `TR`.

## Project Integration Notes

The new sequence generator follows the same public interface pattern as the
existing sequence files:

- `plot`
- `test_report`
- `write_seq`
- `seq_filename`
- imaging and timing parameters as keyword arguments

Like the existing PyPulseq-based generators, the sequence:

- returns a `pypulseq.Sequence` object
- calls `seq.check_timing()`
- optionally prints `seq.test_report()`
- sets `FOV` and `Name` definitions

## PyPulseq Source Review Summary

Before implementation, the following local PyPulseq modules were reviewed to
avoid API assumptions and keep the code aligned with the library behavior:

- `pypulseq/Sequence/sequence.py`
- `pypulseq/Sequence/block.py`
- `pypulseq/make_sinc_pulse.py`
- `pypulseq/make_block_pulse.py`
- `pypulseq/make_trapezoid.py`
- `pypulseq/make_adc.py`
- `pypulseq/calc_duration.py`
- `pypulseq/calc_rf_center.py`
- `pypulseq/check_timing.py`
- `pypulseq/opts.py`

The main implementation decisions derived from that review are:

- RF excitation and refocusing events must use `use='excitation'` and
  `use='refocusing'` so `Sequence.rf_times()` and `Sequence.calculate_kspace()`
  treat them correctly.
- `Sequence.get_block()` exposes decompressed block fields
  `block_duration`, `rf`, `gx`, `gy`, `gz`, `adc`, `label`, `soft_delay`,
  which is consistent with the simulator design already documented in
  `docs/simulation_framework.md`.
- Gradient amplitudes are internally stored in `Hz/m`, while `Opts` can accept
  scanner-style inputs in `mT/m` and `T/m/s`.
- Timing validation is based on `calc_duration()` plus raster alignment checks
  performed by `check_timing()`.
- RF centers must be computed with `calc_rf_center()` rather than inferred from
  raw durations.

## Sequence Design Details

`write_se_sequence()` creates:

- a slice-selective sinc excitation pulse
- a slice-selective sinc refocusing pulse with `phase_offset = pi / 2`
  to satisfy the standard CPMG-style phase relationship
- Cartesian readout and phase-encoding gradients
- prephasers for:
  - readout centering
  - phase encoding
  - slice rephasing after the excitation pulse

Timing is not hard-coded.
Instead, the sequence uses `calc_duration()` and `calc_rf_center()` to compute:

- the minimum feasible `TE`
- the delay before the 180 degree pulse
- the delay between the 180 degree pulse and the readout
- the `TR` fill delay

If the requested `TE` or `TR` is not physically feasible for the generated
events, the function raises a `ValueError` with the minimum valid value.

## Interface Summary

Key parameters:

- `fov`
- `n_x`
- `n_y`
- `slice_thickness`
- `n_slices`
- `tr`
- `te`
- `excitation_flip_angle_deg`
- `refocusing_flip_angle_deg`
- `rf_excitation_duration`
- `rf_refocusing_duration`
- `readout_time`
- `prephase_duration`

Returned value:

- `pypulseq.Sequence`

## Validation

Automated validation lives in `tests/test_spin_echo_sequence.py` and covers:

- timing success via `seq.check_timing()`
- correct excitation, refocusing and ADC counts
- block routing consistency with the simulator
- k-space sampling size and finiteness
- rejection of impossible `TE` values
- a small end-to-end forward simulation and FFT reconstruction

Recommended command:

```bash
python -m unittest discover -s tests -v
```

## Current Scope

This implementation targets a standard single-shot-per-TR Cartesian spin-echo
acquisition pattern and acquires slices sequentially.
That keeps the sequence simple and compatible with the current project
simulator, while staying close to the existing `se1.py` reference logic.
