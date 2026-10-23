import mdtraj as md
import MDAnalysis as mda
import pandas as pd
import numpy as np
import sys
import os
import scipy.stats
import argparse

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")

args = parser.parse_args()

pdb = args.pdb
protname = args.name
save_dir = args.save_dir

def find_xtc_filenames( path_to_dir, suffix=".xtc" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def find_npy_filenames( path_to_dir, suffix=".npy" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# functions for order parameters
def nh_bond_vectors(u, selN='name N and not resname PRO', selH='(name H or name HN) and not resname PRO'):
    """
    Compute unit N–H bond vectors for all residues across the trajectory.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Loaded trajectory.
    selN, selH : str
        Atom selection strings for N and H atoms.

    Returns
    -------
    nh_vectors : np.ndarray, shape (n_frames, n_bonds, 3)
        Unit N–H bond vectors per frame.
    resids : np.ndarray, shape (n_bonds,)
        Residue IDs corresponding to each bond.
    """
    # exclude first residue
    first_resid = u.select_atoms('name CA').resids[0]

    selN = u.select_atoms(selN + f" and not resid {first_resid}")
    selH = u.select_atoms(selH + f" and not resid {first_resid}")

    if len(selN) != len(selH):
        raise ValueError(f"Selections mismatch: {len(selN)} N vs {len(selH)} H atoms")

    n_frames = len(u.trajectory)
    n_bonds = len(selN)
    nh_vectors = np.zeros((n_frames, n_bonds, 3), dtype=np.float32)

    for iframe, ts in enumerate(u.trajectory):
        # difference: H - N (direction from N to H)
        vecs = selH.positions - selN.positions
        # normalize each bond vector
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        nh_vectors[iframe] = vecs

    return nh_vectors, selN.resids

def compute_order_parameters(nh_vectors, weights=None, scaling_factor=1.0):
    """
    Compute amide order parameters (S²) from N–H bond unit vectors.

    Parameters
    ----------
    nh_vectors : np.ndarray, shape (n_frames, n_bonds, 3)
        Unit N–H bond vectors over time.
    weights : np.ndarray, shape (n_frames,), optional
        Optional frame weights (e.g., from reweighting). If None, uniform weights are used.

    Returns
    -------
    S2 : np.ndarray, shape (n_bonds,)
        Order parameter per N–H bond.
    Q : np.ndarray, shape (n_bonds, 3, 3)
        Orientation tensors (⟨v_iα v_iβ⟩ for each bond)
    """
    n_frames, n_bonds, _ = nh_vectors.shape
    if weights is None:
        weights = np.ones(n_frames, dtype=np.float64) / n_frames
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= np.sum(weights)  # normalize

    # weighted mean of outer products for each bond
    # <v v^T> per bond
    Q = np.einsum('t, tij, tik -> ijk', weights, nh_vectors, nh_vectors)

    # compute S^2 = 1.5 * sum(Q^2) - 0.5
    S2 = 1.5 * np.sum(Q**2, axis=(1,2)) - 0.5

    # clip numerical noise
    S2 = np.clip(S2, 0.0, 1.0)

    return S2 * scaling_factor

### PERFORM FORWARD MODEL CALCULATIONS ###

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 
time_factors = [0.5, 1, 2, 3, 5, 10]

# find TAUR from info file
info = pd.read_csv(f'../../ORTHOGONAL_EXP_DATA/{protname}/info.csv')
TAUR = float(info[info['Experiment']=='S2_NH']['S2_iso_tauc'])

# parameters for time-averaging / MD data
frame_dt_ps = 100.0

for traj in trajs:
    ###
    # calculate order parameters using 0.5 x , 1 x , 2 x, 3 x, 5 x, 10 x the value of TAUR for time-averaging blocks
    ###
    trj = mda.Universe(pdb, traj)
    nframes = len(trj.trajectory)

    print('calculating amide bond vectors/S2 for {}'.format(traj))
    if traj==trajs[0]:
        # run NH bond vector calculation
        vec, S2resids = nh_bond_vectors(trj)

    else:
        # run NH bond vector calculation
        tmpvec, S2resids = nh_bond_vectors(trj)
        # append
        vec = np.concatenate((vec, tmpvec),axis=0)

    # frames from NH bond vector data
    n_frames, n_bonds, _ = vec.shape


# compute order parameters using time-averaging of blocks
for time_factor in time_factors:
    print(f'Time averaging order parameters with window size {time_factor} x TAUR')

    window_ns = TAUR * time_factor
    # convert ns -> frames (frame_dt is in ps)
    frames_per_window = max(1, int(round(window_ns * 1000.0 / float(frame_dt_ps))))
    starts = list(range(0, n_frames - frames_per_window + 1, frames_per_window))

    S2_block_frames = []

    for iw, ws in enumerate(starts):
        we = ws + frames_per_window  # exclusive
        block = vec[ws:we]   # shape (fw, n_bonds, 3)
        S2 = compute_order_parameters(block)
        S2_block_frames.append(S2)

    S2_block_frames = np.array(S2_block_frames)
    # save S2 block data
    np.save(save_dir+f'S2_blocks_{time_factor}xTAUR.npy', S2_block_frames)
    np.savetxt(save_dir+'S2_residues.txt', S2resids)
