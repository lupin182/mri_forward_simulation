"""Generate MRiLab compatible phantom and sequence data for comparison experiments."""

import sys
from pathlib import Path

# Add parent directory to path to import from phantom and Sequence
sys.path.insert(0, str(Path(__file__).parent.parent))

from transform.mrilab_phantom import export_phantom_to_mrilab_mat, generate_simple_asymmetric_phantom
from transform.mrilab_sequence import export_sequence_profile_to_mrilab


def main():
    """Generate and export MRiLab compatible data."""
    print("===== Generating MRiLab Compatible Data =====")
    
    # Phantom parameters
    Nz = 2
    Nx = 90
    Ny = 108
    fov_x = 0.18
    fov_y = 0.216
    slice_thickness = 0.002
    
    # Sequence parameters
    flip_angle_deg = 30.0
    tr = 100e-3
    te = 20e-3
    n_slices = 2
    
    transform_dir = Path(__file__).parent
    
    # Generate and export phantom
    print(f"\n1. Generating phantom: Nz={Nz}, Nx={Nx}, Ny={Ny}, FOV={fov_x}x{fov_y}m")
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=Nz, Nx=Nx, Ny=Ny)
    
    phantom_output_path = transform_dir / "phantom_mrilab.mat"
    export_phantom_to_mrilab_mat(
        phantom_output_path,
        rho=rho,
        t1=t1,
        t2=t2,
        fov_x=fov_x,
        fov_y=fov_y,
        slice_thickness=slice_thickness,
        b0_t=3.0
    )
    print(f"   Phantom exported to: {phantom_output_path}")
    
    # Generate and export sequence
    print(f"\n2. Generating GRE sequence: FA={flip_angle_deg}°, TR={tr*1000:.1f}ms, TE={te*1000:.1f}ms")
    sequence_output_dir = transform_dir / "PSD_GRE_LABEL"
    export_sequence_profile_to_mrilab(
        profile="gre_label",
        output_dir=sequence_output_dir,
        psd_name="PSD_GRE_LABEL",
        n_y=Ny,
        n_x=Nx,
        n_slices=n_slices,
        fov=(fov_x, fov_y),
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        te=te,
        slice_thickness=slice_thickness,
        dummy_scans=0,
        ideal_spoiling_reset=True,
        b0_t=3.0
    )
    print(f"   Sequence exported to: {sequence_output_dir}/")
    
    print("\n===== Generation Complete =====")
    print("\nGenerated files:")
    print(f"  - Phantom: {phantom_output_path}")
    print(f"  - Sequence bundle: {sequence_output_dir}/")


if __name__ == "__main__":
    main()
