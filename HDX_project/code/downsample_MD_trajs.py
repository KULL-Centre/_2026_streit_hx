import os
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from tqdm import tqdm

def downsample_xtc(input_path, output_path, stride):
    """Downsample trajectory by a given stride and save to output_path."""
    u = mda.Universe(input_path)
    with XTCWriter(output_path, n_atoms=u.atoms.n_atoms) as w:
        for ts in tqdm(u.trajectory[::stride], desc=f"Downsampling {os.path.basename(input_path)}"):
            w.write(u)

def main(directory):
    xtc_files = [f for f in os.listdir(directory) if f.endswith('.xtc')]
    if not xtc_files:
        print("No .xtc files found.")
        return

    for filename in xtc_files:
        filepath = os.path.join(directory, filename)
        print(f"\nProcessing {filename} ...")
        
        # Load trajectory
        u = mda.Universe(filepath)
        n_frames = len(u.trajectory)
        print(f"  Frames: {n_frames}")
        
        # Check and downsample if needed
        if n_frames == 200001:
            output_file = os.path.join(directory, filename.replace("traj", "downs_traj"))
            stride = n_frames // 20000  # → 10
            print(f"  Downsampling to 20,000 frames (stride={stride})...")
            downsample_xtc(filepath, output_file, stride)
            print(f"  Saved: {output_file}")
        else:
            print("  Skipping (frame count != 200,000).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Downsample 200,001-frame XTC trajectories to 20,000.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing .xtc files (default: current).")
    args = parser.parse_args()
    main(args.directory)
