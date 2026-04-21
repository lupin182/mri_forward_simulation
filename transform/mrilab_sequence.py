"""Export project PyPulseq sequences to MRiLab PSD bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pypulseq as pp
from scipy.io import savemat

from utils import get_rf_frequency_offset_hz, get_rf_phase_offset_rad

# Get correct paths relative to this script's location
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MRILAB_DIR = _PROJECT_ROOT / "MRiLab-1.3"
_MACRO_DIR = _MRILAB_DIR / "Macro"

# Templates are not needed - we'll create XML from scratch
# _SIMU_ATTR_TEMPLATE = None
_DUMMY_PULSE_TEMPLATE = _MACRO_DIR / "SpecialTech" / "DummyPulse.xml"
_EPI_TEMPLATE = _MACRO_DIR / "SpecialTech" / "EPI.xml"


@dataclass(slots=True)
class MRiLabSequenceExport:
    """Summary of an exported MRiLab PSD bundle."""

    bundle_dir: Path
    psd_xml_path: Path
    simu_attr_path: Path
    waveform_paths: dict[str, Path]
    special_paths: dict[str, Path]
    tr_count: int
    regular_tr_count: int


@dataclass(slots=True)
class _SequenceProfile:
    name: str
    sequence: pp.Sequence
    trajectory: str
    n_x: int
    n_y: int
    n_slices: int
    flip_angle_deg: float
    tr_s: float | None
    te_s: float | None
    dummy_trs: int
    reverse_k: bool
    ideal_spoiling: bool


def _ensure_psd_name(name: str) -> str:
    return name if name.startswith("PSD_") else f"PSD_{name}"


def _set_attr(element: ET.Element, **updates: str) -> None:
    for key, value in updates.items():
        element.set(key, value)


def _is_excitation_block(block: Any) -> bool:
    rf = getattr(block, "rf", None)
    return rf is not None and getattr(rf, "use", "undefined") in ("excitation", "undefined")


def _is_refocusing_block(block: Any) -> bool:
    rf = getattr(block, "rf", None)
    return rf is not None and getattr(rf, "use", "undefined") == "refocusing"


def _rf_center_time_s(rf: Any) -> float:
    return float(getattr(rf, "delay", 0.0) + pp.calc_rf_center(rf)[0])


def _split_sequence_into_trs(sequence: pp.Sequence) -> list[list[Any]]:
    trs: list[list[Any]] = []
    prelude: list[Any] = []
    current: list[Any] = []
    seen_excitation = False

    for block_idx in range(1, len(sequence.block_durations) + 1):
        block = sequence.get_block(block_idx)
        if _is_excitation_block(block):
            if seen_excitation:
                trs.append(current)
                current = []
            seen_excitation = True
        if not seen_excitation:
            prelude.append(block)
            continue
        if prelude:
            current.extend(prelude)
            prelude = []
        current.append(block)

    if current:
        trs.append(current)
    elif prelude:
        trs.append(prelude)

    if not trs:
        raise ValueError("Unable to split the sequence into TR blocks.")
    return trs


def _dedupe_waveform(times: np.ndarray, *channels: np.ndarray) -> tuple[np.ndarray, ...]:
    if times.size == 0:
        return (times, *channels)

    order = np.argsort(times, kind="mergesort")
    times = np.asarray(times[order], dtype=np.float64)
    ordered_channels = [np.asarray(channel)[order] for channel in channels]

    unique_times = []
    unique_channels = [[] for _ in ordered_channels]
    idx = 0
    while idx < times.size:
        next_idx = idx
        while next_idx + 1 < times.size and np.isclose(times[next_idx + 1], times[idx], atol=1e-15, rtol=0.0):
            next_idx += 1
        unique_times.append(times[next_idx])
        for out, channel in zip(unique_channels, ordered_channels):
            out.append(channel[next_idx])
        idx = next_idx + 1

    result = [np.asarray(unique_times, dtype=np.float64)]
    result.extend(np.asarray(channel) for channel in unique_channels)
    return tuple(result)


def _pad_waveform(times: np.ndarray, *channels: np.ndarray, row_duration_s: float) -> tuple[np.ndarray, ...]:
    if times.size == 0:
        result = [np.array([0.0, row_duration_s], dtype=np.float64)]
        for channel in channels:
            result.append(np.zeros(2, dtype=np.asarray(channel).dtype if np.asarray(channel).size else np.float64))
        return tuple(result)

    times = np.asarray(times, dtype=np.float64)
    channel_arrays = [np.asarray(channel).copy() for channel in channels]
    if times[0] > 0.0:
        times = np.concatenate(([0.0], times))
        channel_arrays = [np.concatenate(([0], channel)) for channel in channel_arrays]
    if times[-1] < row_duration_s:
        times = np.concatenate((times, [row_duration_s]))
        channel_arrays = [np.concatenate((channel, [0])) for channel in channel_arrays]
    else:
        channel_arrays = [channel.copy() for channel in channel_arrays]
        for channel in channel_arrays:
            channel[-1] = 0
    for channel in channel_arrays:
        channel[0] = 0
    return (times, *channel_arrays)


def _gradient_events(block_time_s: float, grad: Any, gamma_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if grad is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    if grad.type == "trap":
        times = np.array(
            [
                float(grad.delay),
                float(grad.delay + grad.rise_time),
                float(grad.delay + grad.rise_time + grad.flat_time),
                float(grad.delay + grad.rise_time + grad.flat_time + grad.fall_time),
            ],
            dtype=np.float64,
        )
        amps = np.array([0.0, grad.amplitude, grad.amplitude, 0.0], dtype=np.float64) / gamma_hz
    elif grad.type == "grad":
        sample_times = np.asarray(getattr(grad, "tt", getattr(grad, "t", None)), dtype=np.float64)
        if sample_times.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        times = float(grad.delay) + sample_times
        amps = np.asarray(grad.waveform, dtype=np.float64) / gamma_hz
    else:
        raise ValueError(f"Unsupported gradient type: {grad.type}")

    return block_time_s + times, amps


def _rf_events(block_time_s: float, block: Any, gamma_hz: float, system_b0_t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rf = getattr(block, "rf", None)
    if rf is None:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty

    signal = np.asarray(rf.signal, dtype=np.complex128)
    times = block_time_s + float(getattr(rf, "delay", 0.0)) + np.asarray(rf.t, dtype=np.float64)
    amp_t = np.abs(signal) / gamma_hz
    phase_t = np.angle(signal) + get_rf_phase_offset_rad(rf, gamma_hz=gamma_hz, system_b0_t=system_b0_t)
    freq_t = np.full_like(times, -get_rf_frequency_offset_hz(rf, gamma_hz=gamma_hz, system_b0_t=system_b0_t))
    return times, amp_t, phase_t, freq_t


def _adc_events(block_time_s: float, block: Any) -> tuple[np.ndarray, np.ndarray]:
    adc = getattr(block, "adc", None)
    if adc is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    sample_times = block_time_s + (np.arange(adc.num_samples, dtype=np.float64) + 0.5) * float(adc.dwell) + float(adc.delay)
    return sample_times, np.ones(adc.num_samples, dtype=np.float64)


def _render_tr_waveforms(tr_blocks: list[Any], row_duration_s: float, gamma_hz: float, system_b0_t: float) -> dict[str, np.ndarray]:
    current_time = 0.0
    rf_time_parts: list[np.ndarray] = []
    rf_amp_parts: list[np.ndarray] = []
    rf_phase_parts: list[np.ndarray] = []
    rf_freq_parts: list[np.ndarray] = []
    gx_time_parts: list[np.ndarray] = []
    gx_amp_parts: list[np.ndarray] = []
    gy_time_parts: list[np.ndarray] = []
    gy_amp_parts: list[np.ndarray] = []
    gz_time_parts: list[np.ndarray] = []
    gz_amp_parts: list[np.ndarray] = []
    adc_time_parts: list[np.ndarray] = []
    adc_amp_parts: list[np.ndarray] = []

    for block in tr_blocks:
        rf_time, rf_amp, rf_phase, rf_freq = _rf_events(current_time, block, gamma_hz, system_b0_t)
        if rf_time.size:
            rf_time_parts.append(rf_time)
            rf_amp_parts.append(rf_amp)
            rf_phase_parts.append(rf_phase)
            rf_freq_parts.append(rf_freq)

        for attr_name, time_parts, amp_parts in (
            ("gx", gx_time_parts, gx_amp_parts),
            ("gy", gy_time_parts, gy_amp_parts),
            ("gz", gz_time_parts, gz_amp_parts),
        ):
            grad_time, grad_amp = _gradient_events(current_time, getattr(block, attr_name, None), gamma_hz)
            if grad_time.size:
                time_parts.append(grad_time)
                amp_parts.append(grad_amp)

        adc_time, adc_amp = _adc_events(current_time, block)
        if adc_time.size:
            adc_time_parts.append(adc_time)
            adc_amp_parts.append(adc_amp)

        current_time += float(block.block_duration)

    row_duration_s = max(row_duration_s, current_time)

    def finalize_scalar(time_parts: list[np.ndarray], amp_parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not time_parts:
            return np.array([0.0, row_duration_s], dtype=np.float64), np.zeros(2, dtype=np.float64)
        times = np.concatenate(time_parts)
        amps = np.concatenate(amp_parts)
        times, amps = _dedupe_waveform(times, amps)
        return _pad_waveform(times, amps, row_duration_s=row_duration_s)

    rf_times = np.concatenate(rf_time_parts) if rf_time_parts else np.zeros(0, dtype=np.float64)
    rf_amp = np.concatenate(rf_amp_parts) if rf_amp_parts else np.zeros(0, dtype=np.float64)
    rf_phase = np.concatenate(rf_phase_parts) if rf_phase_parts else np.zeros(0, dtype=np.float64)
    rf_freq = np.concatenate(rf_freq_parts) if rf_freq_parts else np.zeros(0, dtype=np.float64)
    rf_times, rf_amp, rf_phase, rf_freq = _dedupe_waveform(rf_times, rf_amp, rf_phase, rf_freq)
    rf_times, rf_amp, rf_phase, rf_freq = _pad_waveform(rf_times, rf_amp, rf_phase, rf_freq, row_duration_s=row_duration_s)

    gx_time, gx_amp = finalize_scalar(gx_time_parts, gx_amp_parts)
    gy_time, gy_amp = finalize_scalar(gy_time_parts, gy_amp_parts)
    gz_time, gz_amp = finalize_scalar(gz_time_parts, gz_amp_parts)
    adc_time, adc_amp = finalize_scalar(adc_time_parts, adc_amp_parts)

    return {
        "rfTime": rf_times,
        "rfAmp": rf_amp,
        "rfPhase": rf_phase,
        "rfFreq": rf_freq,
        "GxTime": gx_time,
        "GxAmp": gx_amp,
        "GyTime": gy_time,
        "GyAmp": gy_amp,
        "GzTime": gz_time,
        "GzAmp": gz_amp,
        "ADCTime": adc_time,
        "ADCAmp": adc_amp,
        "duration": np.array([row_duration_s], dtype=np.float64),
    }


def _stack_rows(rows: list[dict[str, np.ndarray]], time_key: str, amp_key: str) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(row[time_key].size for row in rows)
    times = np.zeros((len(rows), max_len), dtype=np.float64)
    amps = np.zeros((len(rows), max_len), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        row_times = row[time_key]
        row_amps = row[amp_key]
        row_len = row_times.size
        times[row_idx, :row_len] = row_times
        amps[row_idx, :row_len] = row_amps
        times[row_idx, row_len:] = row_times[-1]
    return amps, times


def _stack_rf_rows(rows: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(row["rfTime"].size for row in rows)
    rf_amp = np.zeros((len(rows), max_len), dtype=np.float64)
    rf_phase = np.zeros((len(rows), max_len), dtype=np.float64)
    rf_freq = np.zeros((len(rows), max_len), dtype=np.float64)
    rf_time = np.zeros((len(rows), max_len), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        row_len = row["rfTime"].size
        rf_amp[row_idx, :row_len] = row["rfAmp"]
        rf_phase[row_idx, :row_len] = row["rfPhase"]
        rf_freq[row_idx, :row_len] = row["rfFreq"]
        rf_time[row_idx, :row_len] = row["rfTime"]
        rf_time[row_idx, row_len:] = row["rfTime"][-1]
    return rf_amp, rf_phase, rf_freq, rf_time


def _row_duration(rows: list[dict[str, np.ndarray]]) -> float:
    return float(max(row["duration"][0] for row in rows))


def _min_positive_step(rows: list[dict[str, np.ndarray]]) -> float:
    min_step = np.inf
    for row in rows:
        for key in ("rfTime", "GxTime", "GyTime", "GzTime", "ADCTime"):
            diffs = np.diff(row[key])
            diffs = diffs[diffs > 1e-12]
            if diffs.size:
                min_step = min(min_step, float(diffs.min()))
    return 4e-6 if not np.isfinite(min_step) else min_step


def _gradient_limits(rows: list[dict[str, np.ndarray]]) -> tuple[float, float]:
    max_grad = 0.0
    max_slew = 0.0
    for row in rows:
        for time_key, amp_key in (("GxTime", "GxAmp"), ("GyTime", "GyAmp"), ("GzTime", "GzAmp")):
            amps = np.abs(row[amp_key])
            max_grad = max(max_grad, float(np.max(amps)))
            diffs = np.diff(row[amp_key])
            dts = np.diff(row[time_key])
            valid = dts > 1e-12
            if np.any(valid):
                max_slew = max(max_slew, float(np.max(np.abs(diffs[valid] / dts[valid]))))
    return max_grad, max_slew


def _estimate_bandwidth_hz(rows: list[dict[str, np.ndarray]]) -> float:
    dwell_values = []
    for row in rows:
        adc_time = row["ADCTime"][row["ADCAmp"] > 0]
        if adc_time.size > 1:
            dwell_values.extend(np.diff(adc_time))
    if not dwell_values:
        return 80e3
    dwell_values = np.asarray(dwell_values, dtype=np.float64)
    return float(1.0 / np.median(dwell_values))


def _find_refocusing_center_s(tr_blocks: list[Any]) -> float | None:
    current_time = 0.0
    for block in tr_blocks:
        if _is_refocusing_block(block):
            return current_time + _rf_center_time_s(block.rf)
        current_time += float(block.block_duration)
    return None


def _copy_and_customize_simu_attr(
    output_path: Path,
    *,
    profile: _SequenceProfile,
    tr_s: float,
    te_s: float,
    min_update_rate_s: float,
    bandwidth_hz: float,
    max_grad_t_per_m: float,
    max_slew_t_per_m_s: float,
    b0_t: float,
) -> Path:
    """Create SimuAttr XML from scratch instead of using template."""
    
    # Create root element
    root = ET.Element("MRiLabSeq")
    
    # Add imaging parameters
    imaging = ET.SubElement(root, "Imaging")
    imaging.set("BandWidth", f"{bandwidth_hz:.12g}")
    if hasattr(profile.sequence, 'definitions') and "FOV" in profile.sequence.definitions:
        fov_val = profile.sequence.definitions['FOV']
        imaging.set("FOVFreq", f"{fov_val[0]:.12g}")
        imaging.set("FOVPhase", f"{fov_val[1]:.12g}")
    else:
        imaging.set("FOVFreq", "0.2")
        imaging.set("FOVPhase", "0.2")
    imaging.set("FlipAng", f"{profile.flip_angle_deg:.12g}")
    imaging.set("FreqDir", "$2'A/P','L/R','S/I'")
    imaging.set("ResFreq", str(int(profile.n_x)))
    imaging.set("ResPhase", str(int(profile.n_y)))
    imaging.set("ScanPlane", "$1'Axial','Sagittal','Coronal'")
    imaging.set("SliceNum", str(int(profile.n_slices)))
    if hasattr(profile.sequence, 'definitions') and "FOV" in profile.sequence.definitions:
        fov_val = profile.sequence.definitions['FOV']
        slice_thick = fov_val[2] / max(profile.n_slices, 1) if max(profile.n_slices, 1) > 0 else 0.003
        imaging.set("SliceThick", f"{slice_thick:.12g}")
    else:
        imaging.set("SliceThick", "3e-3")
    imaging.set("TE", f"{te_s:.12g}")
    imaging.set("TEPerTR", "1")
    imaging.set("TR", f"{tr_s:.12g}")
    
    # Add advanced parameters
    advanced = ET.SubElement(root, "Advanced")
    advanced.set("MasterTxCoil", "1")
    advanced.set("MultiTransmit", "$1'off','on'")
    advanced.set("NEX", "1")
    advanced.set("NoFreqAlias", "$1'off','on'")
    advanced.set("NoPhaseAlias", "$1'off','on'")
    advanced.set("NoSliceAlias", "$1'off','on'")
    advanced.set("Shim", "$1'Auto', 'Manual'")
    advanced.set("TEAnchor", "$2'Start', 'Middle', 'End'")
    
    # Add hardware parameters
    hardware = ET.SubElement(root, "Hardware")
    hardware.set("B0", f"{b0_t:.12g}")
    hardware.set("B1Level", "1e-6")
    hardware.set("E1Level", "1e-6")
    hardware.set("MaxGrad", f"{max(0.05, 1.1 * max_grad_t_per_m):.12g}")
    hardware.set("MaxSlewRate", f"{max(200.0, 1.1 * max_slew_t_per_m_s):.12g}")
    hardware.set("MinUpdRate", f"{max(1e-8, min_update_rate_s / 2.0):.12g}")
    hardware.set("Model", "PyPulseq Export")
    hardware.set("NoiseLevel", "10")
    hardware.set("SpinPerVoxel", "1")
    
    # Add recon parameters
    recon = ET.SubElement(root, "Recon")
    recon.set("AutoRecon", "$2'off','on'")
    recon.set("ExternalEng", "")
    recon.set("OutputType", "$1'MAT','ISMRMRD','Both'")
    recon.set("ReconEng", "$1'Default','External'")
    recon.set("ReconType", "$1'Cartesian','NonCart'")
    
    # Write to file
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _write_dummy_pulse_xml(output_path: Path, dummy_trs: int, flip_angle_deg: float, dummy_tr_s: float) -> Path:
    tree = ET.parse(_DUMMY_PULSE_TEMPLATE)
    root = tree.getroot()
    _set_attr(
        root,
        DP_Flag="$2'off','on'",
        DP_FlipAng=f"{flip_angle_deg:.12g}",
        DP_Num=str(int(dummy_trs)),
        DP_TR=f"{dummy_tr_s:.12g}",
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _write_epi_xml(output_path: Path, *, etl: int, shot_num: int) -> Path:
    tree = ET.parse(_EPI_TEMPLATE)
    root = tree.getroot()
    _set_attr(
        root,
        EPI_ETL=str(int(etl)),
        EPI_ESP="1e-3",
        EPI_ShotNum=str(int(shot_num)),
        EPI_EchoShifting="$1'on','off'",
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _build_psd_xml(
    output_path: Path,
    *,
    psd_name: str,
    trajectory: str,
    use_dummy_pulse: bool,
    reverse_k_time_s: float | None,
    ideal_spoiling: bool,
) -> Path:
    root = ET.Element("MRiLabSeq", Name=psd_name, Notes="PyPulseq waveform export")
    ET.SubElement(root, "CVs", **{f"CV{i}": "0" for i in range(1, 15)})
    specials_attrs: dict[str, str] = {}
    if use_dummy_pulse:
        specials_attrs["DummyPulse"] = "^1"
    if trajectory == "epi":
        specials_attrs["EPI"] = "^1"
    ET.SubElement(root, "Specials", **specials_attrs)

    pulses = ET.SubElement(
        root,
        "Pulses",
        Freq="1",
        Notes="exported TR waveform",
        Switch="$1'on','off'",
        TREnd="Inf",
        TRStart="1",
        tE="VCtl.TR",
        tS="0",
    )
    rf_parent = ET.SubElement(pulses, "rf")
    ET.SubElement(
        rf_parent,
        "rfUser",
        AnchorTE="$2'on','off'",
        CoilID="1",
        DupSpacing="0",
        Duplicates="1",
        Notes="PyPulseq RF waveform",
        Switch="$1'on','off'",
        rfFile="'rf.mat'",
    )
    gz_parent = ET.SubElement(pulses, "GzSS")
    ET.SubElement(
        gz_parent,
        "GzUser",
        DupSpacing="0",
        Duplicates="1",
        GzFile="'gz.mat'",
        Notes="PyPulseq Gz waveform",
        Switch="$1'on','off'",
    )
    gy_parent = ET.SubElement(pulses, "GyPE")
    ET.SubElement(
        gy_parent,
        "GyUser",
        DupSpacing="0",
        Duplicates="1",
        GyFile="'gy.mat'",
        Notes="PyPulseq Gy waveform",
        Switch="$1'on','off'",
    )
    gx_parent = ET.SubElement(pulses, "GxR")
    ET.SubElement(
        gx_parent,
        "GxUser",
        DupSpacing="0",
        Duplicates="1",
        GxFile="'gx.mat'",
        Notes="PyPulseq Gx waveform",
        Switch="$1'on','off'",
    )
    adc_parent = ET.SubElement(pulses, "ADC")
    ET.SubElement(
        adc_parent,
        "ADCUser",
        ADCFile="'adc.mat'",
        DupSpacing="0",
        Duplicates="1",
        Notes="PyPulseq ADC waveform",
        Switch="$1'on','off'",
    )
    ext_parent = ET.SubElement(pulses, "Ext")
    ET.SubElement(
        ext_parent,
        "ExtBit",
        DupSpacing="0",
        Duplicates="1",
        Ext="5",
        Notes="calculate remaining scan time",
        Switch="$1'on','off'",
        tStart="0",
    )
    ET.SubElement(
        ext_parent,
        "ExtBit",
        DupSpacing="0",
        Duplicates="1",
        Ext="1",
        Notes="reset K space location",
        Switch="$1'on','off'",
        tStart="10e-6",
    )
    if reverse_k_time_s is not None:
        ET.SubElement(
            ext_parent,
            "ExtBit",
            DupSpacing="0",
            Duplicates="1",
            Ext="2",
            Notes="reverse K space location",
            Switch="$1'on','off'",
            tStart=f"{reverse_k_time_s:.12g}",
        )
    if ideal_spoiling:
        ET.SubElement(
            ext_parent,
            "ExtBit",
            DupSpacing="0",
            Duplicates="1",
            Ext="6",
            Notes="dephase Mxy",
            Switch="$1'on','off'",
            tStart="VCtl.TR-10e-6",
        )

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _write_memo(output_path: Path, profile: _SequenceProfile) -> Path:
    lines = [
        "======PyPulseq to MRiLab Export======",
        f"Profile : {profile.name}",
        "Waveform mode : rfUser/GxUser/GyUser/GzUser/ADCUser",
        "Notes:",
        "- This bundle replays the exported PyPulseq waveforms inside MRiLab.",
        "- Labels, triggers and other non-simulation metadata are not exported.",
        "- MRiLab still applies its own loop bookkeeping via SimuAttr/SpecialTech.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _build_profile(profile: str, **sequence_kwargs: Any) -> _SequenceProfile:
    if profile == "gre":
        from Sequence.write_gre import write_gre_sequence

        seq = write_gre_sequence(plot=False, test_report=False, write_seq=False, **sequence_kwargs)
        return _SequenceProfile(
            name=profile,
            sequence=seq,
            trajectory="cartesian",
            n_x=int(sequence_kwargs.get("n_x", 64)),
            n_y=int(sequence_kwargs.get("n_y", 64)),
            n_slices=1,
            flip_angle_deg=float(sequence_kwargs.get("flip_angle_deg", 10.0)),
            tr_s=float(sequence_kwargs.get("tr", 12e-3)),
            te_s=float(sequence_kwargs.get("te", 5e-3)),
            dummy_trs=int(sequence_kwargs.get("dummy_scans", 0)),
            reverse_k=False,
            ideal_spoiling=bool(sequence_kwargs.get("ideal_spoiling_reset", False)),
        )

    if profile == "gre_label":
        from Sequence.write_gre_label import write_gre_label_sequence

        n_slices = int(sequence_kwargs.get("n_slices", 1))
        dummy_scans = int(sequence_kwargs.get("dummy_scans", 0))
        if dummy_scans and n_slices != 1:
            raise NotImplementedError(
                "MRiLab dummy-pulse bookkeeping only supports the GRE-label export "
                "when dummy scans appear once at the beginning of the scan "
                "(for example n_slices=1)."
            )
        seq = write_gre_label_sequence(plot=False, test_report=False, write_seq=False, **sequence_kwargs)
        return _SequenceProfile(
            name=profile,
            sequence=seq,
            trajectory="cartesian",
            n_x=int(sequence_kwargs.get("n_x", 64)),
            n_y=int(sequence_kwargs.get("n_y", 64 if sequence_kwargs.get("n_y") is not None else sequence_kwargs.get("n_x", 64))),
            n_slices=n_slices,
            flip_angle_deg=float(sequence_kwargs.get("flip_angle_deg", 30.0)),
            tr_s=float(sequence_kwargs.get("tr", 10e-3)),
            te_s=float(sequence_kwargs.get("te", 4.3e-3)),
            dummy_trs=dummy_scans,
            reverse_k=False,
            ideal_spoiling=bool(sequence_kwargs.get("ideal_spoiling_reset", True)),
        )

    if profile == "se":
        from Sequence.write_se import write_se_sequence

        seq = write_se_sequence(plot=False, test_report=False, write_seq=False, **sequence_kwargs)
        return _SequenceProfile(
            name=profile,
            sequence=seq,
            trajectory="cartesian",
            n_x=int(sequence_kwargs.get("n_x", 64)),
            n_y=int(sequence_kwargs.get("n_y", 64)),
            n_slices=int(sequence_kwargs.get("n_slices", 1)),
            flip_angle_deg=float(sequence_kwargs.get("excitation_flip_angle_deg", 90.0)),
            tr_s=float(sequence_kwargs.get("tr", 1.0)),
            te_s=float(sequence_kwargs.get("te", 20e-3)),
            dummy_trs=0,
            reverse_k=True,
            ideal_spoiling=False,
        )

    if profile == "epi":
        from Sequence.write_epi import write_epi_sequence

        seq = write_epi_sequence(plot=False, test_report=False, write_seq=False, **sequence_kwargs)
        return _SequenceProfile(
            name=profile,
            sequence=seq,
            trajectory="epi",
            n_x=int(sequence_kwargs.get("n_x", 64)),
            n_y=int(sequence_kwargs.get("n_y", 64)),
            n_slices=int(sequence_kwargs.get("n_slices", 1)),
            flip_angle_deg=90.0,
            tr_s=None,
            te_s=None,
            dummy_trs=0,
            reverse_k=False,
            ideal_spoiling=False,
        )

    if profile == "epi_se":
        from Sequence.write_epi_se import write_epi_se_sequence

        seq = write_epi_se_sequence(plot=False, test_report=False, write_seq=False, **sequence_kwargs)
        return _SequenceProfile(
            name=profile,
            sequence=seq,
            trajectory="epi",
            n_x=int(sequence_kwargs.get("n_x", 64)),
            n_y=int(sequence_kwargs.get("n_y", 64)),
            n_slices=1,
            flip_angle_deg=90.0,
            tr_s=None,
            te_s=float(sequence_kwargs.get("te", 200e-3)),
            dummy_trs=0,
            reverse_k=True,
            ideal_spoiling=False,
        )

    raise NotImplementedError(
        f"Unsupported export profile '{profile}'. Supported profiles are "
        "'gre', 'gre_label', 'se', 'epi' and 'epi_se'."
    )


def export_sequence_to_mrilab(
    sequence: pp.Sequence,
    output_dir: str | Path,
    *,
    psd_name: str,
    trajectory: str,
    n_x: int,
    n_y: int,
    n_slices: int = 1,
    flip_angle_deg: float = 90.0,
    tr_s: float | None = None,
    te_s: float | None = None,
    dummy_trs: int = 0,
    reverse_k: bool = False,
    ideal_spoiling: bool = False,
    b0_t: float | None = None,
    memo_name: str = "custom",
) -> MRiLabSequenceExport:
    """Export a PyPulseq sequence as a MRiLab PSD bundle using user waveforms."""
    if trajectory not in {"cartesian", "epi"}:
        raise ValueError("trajectory must be either 'cartesian' or 'epi'.")
    if n_y % 2 != 0:
        raise ValueError("MRiLab cartesian/EPI exports require an even phase-encode count.")
    if n_slices != 1 and n_slices % 2 != 0:
        raise ValueError("MRiLab requires SliceNum to be 1 or an even number.")

    psd_name = _ensure_psd_name(psd_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma_hz = float(getattr(sequence.system, "gamma", 42.576e6))
    system_b0_t = float(getattr(sequence.system, "B0", b0_t if b0_t is not None else 3.0))
    if b0_t is None:
        b0_t = system_b0_t

    trs = _split_sequence_into_trs(sequence)
    regular_tr_count = (1 if trajectory == "epi" else n_y) * n_slices
    expected_total_trs = regular_tr_count + dummy_trs
    if len(trs) != expected_total_trs:
        raise ValueError(
            f"Expected {expected_total_trs} TR groups from the provided metadata, "
            f"but found {len(trs)} after splitting the PyPulseq sequence."
        )

    if dummy_trs:
        for tr_blocks in trs[:dummy_trs]:
            if any(getattr(block, "adc", None) is not None for block in tr_blocks):
                raise ValueError("Dummy TRs must not contain ADC samples.")

    row_durations = [sum(float(block.block_duration) for block in tr_blocks) for tr_blocks in trs]
    regular_row_duration = max(row_durations[dummy_trs:] or row_durations)
    dummy_row_duration = max(row_durations[:dummy_trs]) if dummy_trs else regular_row_duration

    rendered_rows = []
    for row_idx, tr_blocks in enumerate(trs):
        row_duration = dummy_row_duration if row_idx < dummy_trs else regular_row_duration
        rendered_rows.append(_render_tr_waveforms(tr_blocks, row_duration, gamma_hz, system_b0_t))

    rf_amp, rf_phase, rf_freq, rf_time = _stack_rf_rows(rendered_rows)
    gx_amp, gx_time = _stack_rows(rendered_rows, "GxTime", "GxAmp")
    gy_amp, gy_time = _stack_rows(rendered_rows, "GyTime", "GyAmp")
    gz_amp, gz_time = _stack_rows(rendered_rows, "GzTime", "GzAmp")
    adc_amp, adc_time = _stack_rows(rendered_rows, "ADCTime", "ADCAmp")

    waveform_paths = {
        "rf": output_dir / "rf.mat",
        "gx": output_dir / "gx.mat",
        "gy": output_dir / "gy.mat",
        "gz": output_dir / "gz.mat",
        "adc": output_dir / "adc.mat",
    }
    savemat(waveform_paths["rf"], {"rfAmp": rf_amp, "rfPhase": rf_phase, "rfFreq": rf_freq, "rfTime": rf_time}, do_compression=True)
    savemat(waveform_paths["gx"], {"GAmp": gx_amp, "GTime": gx_time}, do_compression=True)
    savemat(waveform_paths["gy"], {"GAmp": gy_amp, "GTime": gy_time}, do_compression=True)
    savemat(waveform_paths["gz"], {"GAmp": gz_amp, "GTime": gz_time}, do_compression=True)
    savemat(waveform_paths["adc"], {"GAmp": adc_amp, "GTime": adc_time}, do_compression=True)

    reverse_k_time = _find_refocusing_center_s(trs[dummy_trs]) if reverse_k and len(trs) > dummy_trs else None
    bandwidth_hz = _estimate_bandwidth_hz(rendered_rows[dummy_trs:] or rendered_rows)
    min_step = _min_positive_step(rendered_rows)
    max_grad_t_per_m, max_slew_t_per_m_s = _gradient_limits(rendered_rows)

    tr_value_s = regular_row_duration if tr_s is None else float(tr_s)
    if dummy_trs and dummy_row_duration > tr_value_s:
        tr_value_s = dummy_row_duration
    te_value_s = float(te_s if te_s is not None else tr_value_s / 2.0)

    definitions = getattr(sequence, "definitions", {})
    if "FOV" in definitions:
        fov = np.asarray(definitions["FOV"], dtype=np.float64).reshape(-1)
        fov_x = float(fov[0])
        fov_y = float(fov[1] if fov.size > 1 else fov[0])
        slice_extent = float(fov[2] if fov.size > 2 else 0.003 * max(n_slices, 1))
        slice_thickness = slice_extent / max(n_slices, 1)
    else:
        fov_x = 0.22
        fov_y = 0.22
        slice_thickness = 0.003
    sequence.set_definition("FOV", [fov_x, fov_y, slice_thickness * max(n_slices, 1)])

    profile = _SequenceProfile(
        name=memo_name,
        sequence=sequence,
        trajectory=trajectory,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        flip_angle_deg=flip_angle_deg,
        tr_s=tr_value_s,
        te_s=te_value_s,
        dummy_trs=dummy_trs,
        reverse_k=reverse_k,
        ideal_spoiling=ideal_spoiling,
    )

    special_paths: dict[str, Path] = {}
    simu_attr_path = _copy_and_customize_simu_attr(
        output_dir / "SimuAttr.xml",
        profile=profile,
        tr_s=tr_value_s,
        te_s=te_value_s,
        min_update_rate_s=min_step,
        bandwidth_hz=bandwidth_hz,
        max_grad_t_per_m=max_grad_t_per_m,
        max_slew_t_per_m_s=max_slew_t_per_m_s,
        b0_t=float(b0_t),
    )
    if dummy_trs:
        special_paths["DummyPulse"] = _write_dummy_pulse_xml(
            output_dir / "DummyPulse.xml",
            dummy_trs=dummy_trs,
            flip_angle_deg=flip_angle_deg,
            dummy_tr_s=dummy_row_duration,
        )
    if trajectory == "epi":
        special_paths["EPI"] = _write_epi_xml(output_dir / "EPI.xml", etl=n_y, shot_num=1)

    psd_xml_path = _build_psd_xml(
        output_dir / f"{psd_name}.xml",
        psd_name=psd_name,
        trajectory=trajectory,
        use_dummy_pulse=bool(dummy_trs),
        reverse_k_time_s=reverse_k_time,
        ideal_spoiling=ideal_spoiling,
    )
    _write_memo(output_dir / f"{psd_name}_Memo.txt", profile)

    return MRiLabSequenceExport(
        bundle_dir=output_dir,
        psd_xml_path=psd_xml_path,
        simu_attr_path=simu_attr_path,
        waveform_paths=waveform_paths,
        special_paths=special_paths,
        tr_count=len(trs),
        regular_tr_count=regular_tr_count,
    )


def export_sequence_profile_to_mrilab(
    profile: str,
    output_dir: str | Path,
    *,
    psd_name: str | None = None,
    b0_t: float | None = None,
    **sequence_kwargs: Any,
) -> MRiLabSequenceExport:
    """Build one of this project's sequence profiles and export it to MRiLab."""
    seq_profile = _build_profile(profile, **sequence_kwargs)
    if psd_name is None:
        psd_name = f"PSD_{profile.upper()}"

    return export_sequence_to_mrilab(
        seq_profile.sequence,
        output_dir,
        psd_name=psd_name,
        trajectory=seq_profile.trajectory,
        n_x=seq_profile.n_x,
        n_y=seq_profile.n_y,
        n_slices=seq_profile.n_slices,
        flip_angle_deg=seq_profile.flip_angle_deg,
        tr_s=seq_profile.tr_s,
        te_s=seq_profile.te_s,
        dummy_trs=seq_profile.dummy_trs,
        reverse_k=seq_profile.reverse_k,
        ideal_spoiling=seq_profile.ideal_spoiling,
        b0_t=b0_t,
        memo_name=profile,
    )