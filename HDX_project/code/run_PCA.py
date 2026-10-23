#!/usr/bin/env python3

import argparse
import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA
from utils import *


def get_structured_residues(topology_file, buffer=1):
    """
    Run DSSP on the input reference structure and select non-coil residues.
    Expand the selected regions by +/- buffer residues within each chain.
    """

    ref = md.load(topology_file)
    dssp = md.compute_dssp(ref, simplified=True)[0]

    structured = set()

    for residue, ss in zip(ref.topology.residues, dssp):
        if ss in ("H", "E"):
            structured.add(residue.index)

    expanded = set(structured)

    for chain in ref.topology.chains:
        chain_residues = list(chain.residues)
        chain_indices = [res.index for res in chain_residues]

        for i, res in enumerate(chain_residues):
            if res.index in structured:
                for offset in range(-buffer, buffer + 1):
                    j = i + offset
                    if 0 <= j < len(chain_indices):
                        expanded.add(chain_indices[j])

    return np.array(sorted(expanded), dtype=int)


def ca_ca_distances(traj, selected_residues, min_sep=8):
    """
    Compute CA-CA distances for selected residues separated by at least
    min_sep residues in sequence.
    """

    selected_residues = set(selected_residues)
    ca_atoms = [atom for atom in traj.topology.atoms if atom.name == "CA" and atom.residue.index in selected_residues]

    pairs = []

    for i in range(len(ca_atoms)):
        for j in range(i + 1, len(ca_atoms)):
            res_i = ca_atoms[i].residue
            res_j = ca_atoms[j].residue

            if res_i.chain.index != res_j.chain.index:
                continue

            if abs(res_i.index - res_j.index) >= min_sep:
                pairs.append((ca_atoms[i].index, ca_atoms[j].index))

    pairs = np.asarray(pairs, dtype=int)

    if len(pairs) == 0:
        return np.empty((traj.n_frames, 0))

    return md.compute_distances(traj, pairs)


def contacts_switch(distances, r0=0.8, d0=0.0, n=6, m=10):
    r = np.asarray(distances, dtype=float)
    x = (r - d0) / r0
    num = 1.0 - np.power(x, n)
    den = 1.0 - np.power(x, m)
    out = np.divide(num, den, out=np.full_like(r, n / m), where=np.abs(den) > 1e-12)
    return out


def dihedral_features(traj, selected_residues):
    """
    Calculate sin/cos(phi) and sin/cos(psi) for selected residues.
    """

    selected_residues = set(selected_residues)

    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)

    phi_residues = np.array([traj.topology.atom(indices[1]).residue.index for indices in phi_indices])
    psi_residues = np.array([traj.topology.atom(indices[1]).residue.index for indices in psi_indices])

    phi_mask = np.array([res in selected_residues for res in phi_residues])
    psi_mask = np.array([res in selected_residues for res in psi_residues])

    phi = phi[:,phi_mask]
    psi = psi[:,psi_mask]

    features = np.concatenate((np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)), axis=1)

    return features


def main():

    parser = argparse.ArgumentParser(description="Compute structured-region CA contacts and phi/psi features from all XTC files followed by PCA.")
    parser.add_argument("-t", "--topology", required=True, help="Topology/reference structure file, e.g. PDB")
    parser.add_argument("-o", "--output-prefix", required=True, help="Output file prefix, e.g. PCA/ or run1_")
    parser.add_argument("--min-sep", type=int, default=8, help="Minimum residue separation for contacts (default: 8)")
    parser.add_argument("--dssp-buffer", type=int, default=1, help="Number of residues to include around DSSP-structured regions (default: 1)")
    parser.add_argument("--r0", type=float, default=0.8, help="Switch parameter r0 in nm (default: 0.8)")
    parser.add_argument("--d0", type=float, default=0.0, help="Switch parameter d0 in nm (default: 0.0)")
    parser.add_argument("--n", type=int, default=6, help="Switch numerator exponent (default: 6)")
    parser.add_argument("--m", type=int, default=10, help="Switch denominator exponent (default: 10)")
    parser.add_argument("--dihedral-weight", type=float, default=0.25, help="Weight applied to sin/cos phi/psi descriptors before PCA (default: 0.25)")
    args = parser.parse_args()

    output = args.output_prefix

    # find trajectories using exactly the same logic as the reweighting code
    trajs = sorted(find_xtc_filenames('./'))

    if len(trajs) == 0:
        raise FileNotFoundError("No XTC files found in the current directory.")

    print(f"Found {len(trajs)} trajectories:")
    for traj in trajs:
        print(traj)

    # determine structured residues from reference PDB using DSSP
    structured_residues = get_structured_residues(args.topology, buffer=args.dssp_buffer)

    print(f"Selected {len(structured_residues)} structured/buffer residues:")
    print(structured_residues)

    np.savetxt(output + "selected_residues.txt", structured_residues, fmt="%d")

    all_features = []
    frame_counts = []

    for traj_file in trajs:
        print(f"Processing {traj_file}")

        u = md.load(traj_file, top=args.topology)

        dist = ca_ca_distances(u, structured_residues, min_sep=args.min_sep)

        if dist.shape[1] == 0:
            raise ValueError("No valid CA-CA pairs found for the selected structured residues and min_sep.")

        contacts = contacts_switch(dist, r0=args.r0, d0=args.d0, n=args.n, m=args.m)
        dihedrals = dihedral_features(u, structured_residues)

        # downweight local dihedral descriptors relative to contact descriptors
        dihedrals_weighted = args.dihedral_weight * dihedrals

        # combine contacts and downweighted dihedral descriptors
        features = np.concatenate((contacts, dihedrals_weighted), axis=1)

        all_features.append(features)
        frame_counts.append(u.n_frames)

        print(f"Frames: {u.n_frames}")
        print(f"Contact features: {contacts.shape[1]}")
        print(f"Dihedral features: {dihedrals.shape[1]}")
        print(f"Dihedral weight: {args.dihedral_weight}")
        print(f"Total features: {features.shape[1]}")

    # concatenate descriptors from all trajectories in the same sorted trajectory order
    features = np.concatenate(all_features, axis=0)

    print(f"Total frames: {features.shape[0]}")
    print(f"Total features per frame: {features.shape[1]}")

    # PCA on the combined ensemble
    pca = PCA(n_components=2)
    contact_pca = pca.fit_transform(features)

    explained = pca.explained_variance_ratio_

    with open(output + "pca_explained_variance.txt", "w") as f:
        f.write(f"PC1: {explained[0]:.4f}\n")
        f.write(f"PC2: {explained[1]:.4f}\n")
        f.write(f"Total (PC1+PC2): {explained.sum():.4f}\n")
        f.write(f"Dihedral weight: {args.dihedral_weight:.4f}\n")

    np.savetxt(output + "pca_coordinates.txt", contact_pca, fmt="%.6f")

    with open(output + "trajectory_frames.txt", "w") as f:
        start = 0
        for traj_file, n_frames in zip(trajs, frame_counts):
            end = start + n_frames
            f.write(f"{traj_file} {start} {end} {n_frames}\n")
            start = end

    print("Done.")
    print(f"Saved: {output}selected_residues.txt")
    print(f"Saved: {output}pca_explained_variance.txt")
    print(f"Saved: {output}pca_coordinates.txt")
    print(f"Saved: {output}trajectory_frames.txt")


if __name__ == "__main__":
    main()