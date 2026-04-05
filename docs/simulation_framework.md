# Simulation Framework

## Scope

This repository now contains a block-driven forward MRI simulator that works directly on `pypulseq.Sequence.get_block()` outputs.
The implementation was validated against local `pypulseq 1.5.0.post1`.

## PyPulseq Notes

The simulator relies on the public block namespace returned by `get_block()` instead of internal compressed event IDs.
The block fields observed in the local library are:

- `block_duration`
- `rf`
- `gx`, `gy`, `gz`
- `adc`
- `label`
- `soft_delay`

Relevant unit conventions in the local PyPulseq build:

- RF waveform samples are in `Hz`
- Gradient amplitudes are in `Hz/m`
- ADC sample times are `adc.delay + (n + 0.5) * adc.dwell`

## Main Loop

The entry point is `simulate()` in `simulate.py`.
For each block:

1. Read the decompressed block from `sequence.get_block(block_idx)`.
2. Classify the block.
3. Route RF- or ADC-containing blocks to the fine solver.
4. Route all other blocks to the fast analytic solver.
5. Collect ADC samples when present.

The helper `analyze_sequence_blocks()` exposes the routing decision for tests and debugging.

## Fine Solver

The fine solver is used for blocks containing RF or ADC.
A refined time grid is built from:

- block start and end
- RF sample edges
- gradient breakpoints
- ADC sample times
- additional sub-steps to satisfy `SimulationConfig.fine_dt`

For each sub-step the simulator:

1. Samples RF and gradients at the midpoint.
2. Builds the instantaneous off-resonance term.
3. Calls `apply_bloch_step()` from `bloch_kernel.py`.
4. Records the receive signal at ADC sample instants.

This path is intentionally conservative and favors correctness over aggressive fusion.

## Fast Solver

The fast solver is exact for blocks without RF and ADC.
It uses the closed-form solution of free precession with relaxation:

- `Mxy(t) = Mxy(0) * exp(-t / T2) * exp(-i * phase)`
- `Mz(t) = M0 + (Mz(0) - M0) * exp(-t / T1)`

The accumulated phase is computed from:

- chemical shift
- `dB0`
- random off-resonance `dWRnd`
- integrated gradient areas from the block

Because the phase term depends on gradient area, this path remains exact for trapezoids and arbitrary gradients as long as the block contains no RF.

## Bloch Kernel Adaptation

`bloch_kernel.py` was refactored to provide:

- `build_off_resonance_rad_s()`
- `combine_transmit_field_hz()`
- `apply_bloch_step()`
- backward-compatible `BlochKernel()`

Key fixes:

- RF is now treated in PyPulseq-native `Hz`
- gradients are now treated in PyPulseq-native `Hz/m`
- the RF rotation uses `M x B` sign convention
- the legacy wrapper keeps the old symbol name while forwarding into the new implementation

## Phantom Preprocessing

`preprocess_phantom()` now returns a flattened copy instead of mutating the input object in place.
The transmit coil broadcasting bug was also fixed by using `TxCoilNum` instead of `RxCoilNum` for Tx maps.

## Running Validation

Validated command:

```bash
python -m unittest tests.test_validation_unittest -v
```

## Current Assumptions

- `CS` is interpreted in `Hz`
- `dB0` is interpreted in `Tesla`
- `dWRnd` is interpreted in `rad/s`
- receive coils are combined by direct complex summation
- ADC demodulation uses the local ADC phase/frequency offset terms available in the block
