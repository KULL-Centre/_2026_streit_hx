# Script assumes that the first residue in the reference PDB file is number 1.

import MDAnalysis as mda
import numpy as np
import argparse
import mdtraj as md


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Identify native heavy-atom contacts from a reference PDB and write them to contacts.txt"
    )
    parser.add_argument("-p", "--pdb", required=True, help="Reference PDB file")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.45,
        help="Distance cutoff in nm to define contacts (default: 0.45)",
    )
    parser.add_argument(
        "--min_seq_sep",
        type=int,
        default=4,
        help="Minimum sequence separation between residues (default: 4)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="contacts.txt",
        help="Output contact file (default: contacts.txt)",
    )
    return parser.parse_args()


def identify_contacts(atoms, cutoff, structured_residues, min_seq_sep=4):
    """
    Identify heavy-atom contacts in the reference structure.

    Returns a list of tuples:
    (resid1, resname1, atomname1, resid2, resname2, atomname2, ref_distance_nm)
    """
    positions = atoms.positions / 10.0  # Å -> nm
    n_atoms = len(atoms)
    contacts = []

    structured_residues = set(structured_residues)

    for i in range(n_atoms):
        atom_i = atoms[i]
        for j in range(i + 1, n_atoms):
            atom_j = atoms[j]

            # Sequence separation filter
            if abs(atom_i.resid - atom_j.resid) < min_seq_sep:
                continue

            # Keep only contacts where both residues are structured
            if atom_i.resid not in structured_residues or atom_j.resid not in structured_residues:
                continue

            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                contacts.append(
                    (
                        atom_i.resid,
                        atom_i.resname,
                        atom_i.name,
                        atom_j.resid,
                        atom_j.resname,
                        atom_j.name,
                        dist,
                    )
                )

    return contacts


def main():
    args = parse_arguments()

    # Load reference PDB only
    u = mda.Universe(args.pdb)

    # Heavy atoms only
    heavy_atoms = u.select_atoms("not type H")

    # DSSP on reference structure
    traj = md.load(args.pdb)
    dssp = md.compute_dssp(traj, simplified=True)[0]
    structured_residues = np.where(dssp != "C")[0] + 1  # residue numbering assumed 1-based

    contacts = identify_contacts(
        heavy_atoms,
        cutoff=args.cutoff,
        structured_residues=structured_residues,
        min_seq_sep=args.min_seq_sep,
    )

    if len(contacts) == 0:
        raise ValueError("No contacts found with the given cutoff and sequence separation.")

    with open(args.output, "w") as f:
        for resid1, resname1, atomname1, resid2, resname2, atomname2, dist in contacts:
            f.write(
                f"{resid1} {resname1} {atomname1} "
                f"{resid2} {resname2} {atomname2} "
                f"{dist:.6f}\n"
            )

    print(f"Wrote {len(contacts)} contacts to {args.output}")


if __name__ == "__main__":
    main()
