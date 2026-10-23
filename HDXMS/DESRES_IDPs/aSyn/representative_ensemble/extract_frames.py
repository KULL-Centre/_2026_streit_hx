#!/usr/bin/env python

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np


def read_frames(frame_file):
    """
    Read frame indices from a text file.

    Expected format:
        one frame index per line

    Blank lines and lines starting with # are ignored.
    """
    frames = []
    with open(frame_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            frames.append(int(line))
    return np.array(frames, dtype=int)


def main():
    parser = argparse.ArgumentParser(
        description="Extract selected frames from an XTC trajectory and write them as PDB files."
    )
    parser.add_argument("--pdb", required=True, help="Input topology/reference PDB file")
    parser.add_argument("--xtc", required=True, help="Input XTC trajectory file")
    parser.add_argument(
        "--frames",
        default="frames.txt",
        help="Text file containing frame indices to extract, one per line",
    )
    parser.add_argument(
        "--outdir",
        default="extracted_frames",
        help="Output directory for extracted PDB files",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Prefix for output PDB files",
    )
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="Interpret frame numbers in frames.txt as 1-indexed instead of 0-indexed",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = read_frames(args.frames)

    if args.one_indexed:
        frames = frames - 1

    if np.any(frames < 0):
        raise ValueError("Frame indices must be >= 0 after indexing conversion.")

    print(f"Loading trajectory: {args.xtc}")
    traj = md.load(args.xtc, top=args.pdb)

    n_frames = traj.n_frames
    invalid = frames[frames >= n_frames]
    if len(invalid) > 0:
        raise ValueError(
            f"Requested frame(s) outside trajectory range 0–{n_frames - 1}: "
            f"{invalid.tolist()}"
        )

    for frame in frames:
        out_pdb = outdir / f"{args.prefix}_{frame:06d}.pdb"
        traj[frame].save_pdb(out_pdb)
        print(f"Wrote {out_pdb}")

    print("Done.")


if __name__ == "__main__":
    main()
