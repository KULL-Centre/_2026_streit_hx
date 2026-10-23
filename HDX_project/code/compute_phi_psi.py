#!/usr/bin/env python3
import mdtraj as md
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute phi/psi dihedral angles (radians) and save using np.save"
    )
    parser.add_argument("--pdb", required=True, help="PDB topology file")
    parser.add_argument("--traj", required=True, help="XTC trajectory file")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride when loading trajectory (default=1)")
    parser.add_argument("--out", type=str, default="dihedrals",
                        help="Output filename prefix (default='dihedrals')")
    parser.add_argument("--save_dir", type=str, default="dihedrals",
                        help="Output directory (default='dihedrals')")

    args = parser.parse_args()

    print("Loading trajectory...")
    traj = md.load(args.traj, top=args.pdb, stride=args.stride)

    print("Computing dihedral angles (radians)...")
    indices_phi, phi = md.compute_phi(traj)
    indices_psi, psi = md.compute_psi(traj)
    # labels
    phi_label=[]
    psi_label=[]
    # cycle on the number of phi dihedrals (n_phi)
    for i_phi in range(0, indices_phi.shape[0]):
        resindex=traj.topology.atom(indices_phi[i_phi][3]).residue.resSeq
        phi_label.append(resindex)
    for i_psi in range(0, indices_psi.shape[0]):
        resindex=traj.topology.atom(indices_psi[i_psi][3]).residue.resSeq
        psi_label.append(resindex)
    # convert to numpy array
    phi_label=np.array(phi_label)
    psi_label=np.array(psi_label)
    # Output file names
    phi_file = f"{args.save_dir}/phi_{args.out}.npy"
    psi_file = f"{args.save_dir}/psi_{args.out}.npy"
    phi_res_file = f"{args.save_dir}/phi_resids.txt"
    psi_res_file = f"{args.save_dir}/psi_resids.txt"

    print(f"Saving {phi_file} ...")
    np.save(phi_file, np.round(phi, 3))

    print(f"Saving {psi_file} ...")
    np.save(psi_file, np.round(psi, 3))

    print(f"Saving residue IDs...")
    np.savetxt(phi_res_file, phi_label.astype(int), fmt="%d")
    np.savetxt(psi_res_file, psi_label.astype(int), fmt="%d")

    print("Done.")

if __name__ == "__main__":
    main()
