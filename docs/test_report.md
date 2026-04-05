# Test Report

## Environment

- Working directory: `e:\±œ“µøŒÃ‚\mri_codex`
- Python: local Anaconda interpreter
- PyPulseq: `1.5.0.post1`
- Validated runner: `python -m unittest tests.test_validation_unittest -v`

## Test Cases

### 1. Bloch kernel unit test

Purpose:
Verify that a simple RF pulse rotates longitudinal magnetization into the transverse plane with the expected sign convention.

Result:
Passed.

### 2. Block routing test

Purpose:
Verify that RF and ADC blocks are routed to the fine solver and that label-only blocks stay on the fast solver.

Result:
Passed.

### 3. Fast vs fine consistency test

Purpose:
Compare the analytic free-precession solver against the refined numerical solver on a gradient-only block.

Result:
Passed.
Observed maximum absolute differences were on the order of `1e-14` in the local validation script.

### 4. End-to-end GRE test

Purpose:
Run a complete GRE sequence on a small phantom and verify ADC/sample alignment and non-zero signal output.

Result:
Passed.

## Automated Validation Summary

Command:

```bash
python -m unittest tests.test_validation_unittest -v
```

Observed result:

- `Ran 5 tests`
- `Failures: 0`
- `Errors: 0`
- total wall time measured from a Python wrapper: about `5.80 s`

## Performance Snapshot

Small GRE benchmark configuration:

- phantom: `1 x 8 x 8`
- sequence: `n_x=8`, `n_y=4`
- fine solver step cap: `5e-6 s`

Observed result:

- simulation wall time: about `2.30 s`
- ADC samples: `32`
- fine blocks: `8`
- fast blocks: `12`

## Notes

- The repository also contains pytest-style test files, but the validated local runner for this delivery is the `unittest` suite above.
- Sequence generators still print `Timing check passed successfully` during test execution because they call `seq.check_timing()`.
