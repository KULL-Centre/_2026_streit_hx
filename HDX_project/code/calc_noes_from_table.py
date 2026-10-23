#!/usr/bin/env python3
"""
Calculate NOE effective distances using MDAnalysis from an experimental NOE table.

Input format (whitespace-separated):
res1 atom1 res2 atom2 exp_avg exp_lower_diff exp_upper_diff
Example:
2 HG# 3 HN 3.0 1.2 0.3
74 HN 75 QB 3.18 1 1

Handles wildcards:
- Atom names ending with '#' match all atoms starting with that prefix (HG1, HG2, etc.)
- 'Q' followed by one letter (e.g., QB, QG) matches all protons in that group
  according to name prefix (HB, HG, etc.)
"""

import numpy as np
import MDAnalysis as mda
import sys
from pathlib import Path

def parse_noe_table(filename):
    """Read NOE table and return list of dicts."""
    pairs = []
    restraints = []
    with open(filename) as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            res1, atom1, res2, atom2 = int(parts[0]), parts[1], int(parts[2]), parts[3]
            avg, lower, upper = float(parts[4]), float(parts[5]), float(parts[6])
            exp_vals = [float(x) for x in parts[4:]] if len(parts) > 4 else []
            pairs.append({"res1": res1, "atom1": atom1, "res2": res2, "atom2": atom2, "exp": exp_vals})
            restraints.append([avg, avg-lower, avg+upper])
    return pairs, restraints


def select_equivalent_atoms(u, resid, atom_pattern):
    """
    Return MDAnalysis AtomGroup for an atom pattern.
    Supports:
      HG# -> all atoms starting with HG
      QB  -> all protons in B group (HB1, HB2, ...)
      exact names otherwise
    """
    if atom_pattern.endswith("#"):
        prefix = atom_pattern[:-1]
        sel = f"resid {resid} and name {prefix}*"
    elif atom_pattern.startswith("Q") and len(atom_pattern) == 2:
        prefix = "H" + atom_pattern[1]
        sel = f"resid {resid} and name {prefix}*"
    elif atom_pattern=="HN" or atom_pattern=="H":
        sel = f"resid {resid} and (name HN or name H)"
    else:
        sel = f"resid {resid} and name {atom_pattern}"
    ag = u.select_atoms(sel)
    if len(ag) == 0:
        print(f"Warning: no atoms found for selection '{sel}'")
    return ag


def calc_noes(topfile, trajfile, noefile, outfile="NOE_distances.npy"):
    u = mda.Universe(topfile, trajfile)
    pairs, restraints = parse_noe_table(noefile)

    # save restraint distances
    np.savetxt(outfile.split('/')[0]+'/Exp_NOE_distances.txt', np.round(restraints, 4), fmt='%.2f')

    nframes = len(u.trajectory)
    npairs = len(pairs)
    dist_array = np.zeros((npairs, nframes), dtype=np.float32)

    for ipair, pair in enumerate(pairs):
        g1 = select_equivalent_atoms(u, pair["res1"], pair["atom1"])
        g2 = select_equivalent_atoms(u, pair["res2"], pair["atom2"])

        if len(g1) == 0 or len(g2) == 0:
            continue

        n1, n2 = len(g1), len(g2)
        d_all = np.zeros(nframes)

        for iframe, ts in enumerate(u.trajectory):
            coords1 = g1.positions
            coords2 = g2.positions
            # Compute all pairwise distances between equivalent atoms
            diff = coords1[:, None, :] - coords2[None, :, :]
            dists = np.linalg.norm(diff, axis=-1).ravel()
            # Average over equivalent protons: <r^-6>^-1/6
            d_all[iframe] = (np.mean(dists ** -6)) ** (-1 / 6)
        print('Number of equivalents: ', dists.shape)
        dist_array[ipair] = d_all
        print(f"Processed {ipair+1}/{npairs}: {pair['res1']} {pair['atom1']} - {pair['res2']} {pair['atom2']}")
    dist_array = dist_array.T
    np.save(outfile, dist_array)
    print(f"\nSaved {npairs}×{nframes} NOE distances to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python calc_noes_from_table.py topfile trajfile noefile [outfile]")
        sys.exit(1)

    topfile = sys.argv[1]
    trajfile = sys.argv[2]
    noefile = sys.argv[3]
    outfile = sys.argv[4] if len(sys.argv) > 4 else "NOE_distances.npy"

    calc_noes(topfile, trajfile, noefile, outfile)
