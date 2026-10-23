import argparse
from pathlib import Path
import os
from scipy.spatial.distance import pdist
import mdtraj as md
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--save_dir", default="outputs")
parser.add_argument(
    "--filter_frames",
    action="store_true",
    default=False,
    help="Filter out frames with overlapping atoms; otherwise SASA calculation may fail."
)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--chunk", type=int, default=100)

# distance scale for switching function, in nm
parser.add_argument("--r_cut", type=float, default=0.50)
parser.add_argument("--cutoff_nm", type=float, default=0.60) # for neighbour search, in nm

# angular weighting: max(0, cos(theta)) ** angular_power
parser.add_argument("--angular_power", type=float, default=8.0)

args = parser.parse_args()

Path(args.save_dir).mkdir(parents=True, exist_ok=True)

top = md.load(args.pdb).topology

def find_xtc_filenames( path_to_dir, suffix=".xtc" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# function to filter out frames that have overlapping atoms (SASA calculation will fail otherwise)
def filter_frames(pdb_file, xtc_file, threshold=0.001):
    print(f"Loading trajectory: {xtc_file}")
    traj = md.load(xtc_file, top=pdb_file)
    print(f"Trajectory loaded with {traj.n_frames} frames and {traj.n_atoms} atoms.")

    ok_frames = []
    min_dists = []

    for i, xyz in enumerate(traj.xyz):
        # Compute all pairwise distances (upper triangle only)
        min_dist = pdist(xyz).min()
        min_dists.append(min_dist)
        if min_dist > threshold:
            ok_frames.append(i)
        else:
            print(f"Frame {i}: Minimum distance = {min_dist:.6f} nm (overlap detected)")

    print("\nSummary:")
    print(f"  Threshold: {threshold} Å")
    print(f"  Frames checked: {traj.n_frames}")
    print(f"  Frames OK: {len(ok_frames)}")
    print(f"  Frames failing: {traj.n_frames - len(ok_frames)}")

    print("\nReturning frames passing (min_dist > threshold):")

    return np.array(ok_frames).astype(int)

# -----------------------------
# Amide N-H pairs
# -----------------------------

# More robust than putting all exclusions into the MDTraj selection DSL

'''
candidate_N_idx = top.select("protein and backbone and name N")

N_idx = []
H_idx = []

excluded_resnames = {"PRO", "ACE", "NME", "NH2"}
amide_H_names = {"H", "HN", "H1", "HT1"}

for n in candidate_N_idx:
    res = top.atom(n).residue

    if res.name in excluded_resnames:
        continue

    h = next(
        (a.index for a in res.atoms if a.name in amide_H_names),
        -1
    )

    if h >= 0:
        N_idx.append(n)
        H_idx.append(h)

N_idx = np.asarray(N_idx, dtype=int)
H_idx = np.asarray(H_idx, dtype=int)
'''
N_idx = top.select("protein and name N and not resname PRO and not resname ACE and not resname NME and not resname NH2")
H_idx = top.select("protein and (name HN or name H) and not resname PRO and not resname ACE and not resname NME and not resname NH2")

assert len(N_idx) == len(H_idx), "Number of N and H atoms in amide pairs should be the same"


res_idx = np.array([top.atom(n).residue.index for n in N_idx], dtype=int)
resid = np.array([top.atom(n).residue.resSeq for n in N_idx], dtype=int)
resname = np.array([top.atom(n).residue.name for n in N_idx])
chain_idx = np.array([top.atom(n).residue.chain.index for n in N_idx], dtype=int)


# -----------------------------
# Heavy atoms and exclusions
# -----------------------------

heavy_idx = top.select("protein and not element H")
heavy_res_idx = np.array([top.atom(i).residue.index for i in heavy_idx], dtype=int)
heavy_chain_idx = np.array([top.atom(i).residue.chain.index for i in heavy_idx], dtype=int)

# No exclusions for front-facing density
allowed_heavy_per_amide = []

for ri, ci in zip(res_idx, chain_idx):
    exclude = (heavy_chain_idx == ci) & (np.abs(heavy_res_idx - ri) <= -1) # no exclusions
    allowed = heavy_idx[~exclude]
    allowed_heavy_per_amide.append(allowed)


# -----------------------------
# Weighting functions
# -----------------------------

def switching(r, r_cut):
    """
    Smooth contact / distance weight.

    """
    x = r / r_cut
    return 1.0 / (1.0 + x**12)


def angular_weight(cos_theta, angular_power):
    """
    Directional/front-side angular weight.

    cos_theta is the cosine of the angle between N->H and H->heavy_atom.
    Only atoms in front of the amide H contribute.
    """
    return np.maximum(0.0, cos_theta) ** angular_power


# -----------------------------
# Stream trajectories
# -----------------------------

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 

for i, traj in enumerate(trajs):

    if args.filter_frames:
        print('Filtering frames for trajectory:', traj)
        ok_frames = filter_frames(args.pdb, traj)
        if len(ok_frames) == 0:
            print(f"No valid frames found in trajectory: {traj}. Skipping.")
            continue
        np.savetxt(args.save_dir+f'okframes_{traj[:-4]}.txt', ok_frames, fmt='%d')

    print('Calculating rho for trajectory:', traj)

    rho_chunks = []

    for chunk in md.iterload(
        traj,
        top=args.pdb,
        stride=args.stride,
        chunk=args.chunk,
    ):
        xyz = chunk.xyz
        n_frames = chunk.n_frames
        n_amides = len(N_idx)

        rho = np.zeros((n_frames, n_amides), dtype=np.float32)

        for k, (N, H, allowed_heavy) in enumerate(
            zip(N_idx, H_idx, allowed_heavy_per_amide)
        ):
            if len(allowed_heavy) == 0:
                continue

            # Neighbors within r_cut of this one amide H, returned per frame.
            neighbours = md.compute_neighbors(
                chunk,
                cutoff=args.cutoff_nm,
                query_indices=np.array([H], dtype=int),
                haystack_indices=allowed_heavy,
            )

            for f, nbrs in enumerate(neighbours):
                if len(nbrs) == 0:
                    continue

                # Unit vector N -> H
                NH = xyz[f, H] - xyz[f, N]
                NH_norm = np.linalg.norm(NH)

                if NH_norm < 1e-8:
                    continue

                NH /= NH_norm

                # Vectors H -> heavy_atom_j
                HV = xyz[f, nbrs] - xyz[f, H]
                dist = np.linalg.norm(HV, axis=1)

                valid = dist > 1e-8
                if not np.any(valid):
                    continue

                HV = HV[valid]
                dist = dist[valid]

                cos_theta = (HV @ NH) / dist

                rho[f, k] = np.sum(
                    switching(dist, args.r_cut)
                    * angular_weight(cos_theta, args.angular_power)
                )

        rho_chunks.append(rho)

    if args.filter_frames:
        rho = np.concatenate(rho_chunks, axis=0)[ok_frames]
    else:
        rho = np.concatenate(rho_chunks, axis=0)

    if i==0:
        all_rho = rho
    else:
        all_rho = np.concatenate((all_rho, rho), axis=0)


# -----------------------------
# Save
# -----------------------------

np.save(f"{args.save_dir}/rho_front_frames.npy", all_rho)

np.savetxt(f"{args.save_dir}/amideresidx.txt", res_idx, fmt="%d")
np.savetxt(f"{args.save_dir}/amideresidues.txt", resid, fmt="%d")
np.savetxt(f"{args.save_dir}/amide_N_indices.txt", N_idx, fmt="%d")
np.savetxt(f"{args.save_dir}/amide_H_indices.txt", H_idx, fmt="%d")
np.savetxt(f"{args.save_dir}/amide_resnames.txt", resname, fmt="%s")
np.savetxt(f"{args.save_dir}/amide_chain_indices.txt", chain_idx, fmt="%d")

print(
    f"Frames: {all_rho.shape[0]} | "
    f"Amides: {len(N_idx)} | "
    f"rho shape: {all_rho.shape} | "
    f"r_cut: {args.r_cut:.3f} nm | "
    f"angular_power: {args.angular_power:.1f} | "
    f"Saved to: {args.save_dir}"
)