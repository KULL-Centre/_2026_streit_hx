import MDAnalysis as mda
from MDAnalysis.lib.util import convert_aa_code
import numpy as np
import argparse
import mdtraj as md

# Warning - this script assumes that coordinates of the input files are in Angstrom

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate fraction of native contacts using PLUMED Q switching function. Assumes coordinates are in Angstrom.")
    parser.add_argument("-p", "--pdb", required=True, help="Reference PDB file")
    parser.add_argument("-x", "--xtc", required=True, help="Trajectory XTC file")
    parser.add_argument("--lambda_param", type=float, default=1.8, help="Lambda parameter for Q switching function")
    parser.add_argument("--beta", type=float, default=50.0, help="Beta parameter (nm^-1) for Q switching function")
    parser.add_argument("--cutoff", type=float, default=0.45, help="Distance cutoff (nm) to define native contacts")
    parser.add_argument("-o", "--output", default="native_contacts.txt", help="Output text file")
    return parser.parse_args()

def identify_native_contacts(atoms, cutoff, structured_residues, min_seq_sep=4):
    """Identify native contacts among heavy atoms with at least min_seq_sep residues apart."""
    positions = atoms.positions / 10.0  # Å -> nm
    atom_indices = atoms.indices # 0-indexed
    n_atoms = len(atoms)
    native_contacts = []
    native_contacts_atoms = [] # list of atom pairs 1-indexed like the pdb file
    r0_values = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Sequence separation filter
            if abs(atoms[i].resid - atoms[j].resid) < min_seq_sep:
                continue
            # Avoid residue pairs that are not both part of the structured residue list
            if atoms[i].resid in structured_residues and atoms[j].resid in structured_residues:
                pass
            else:
                continue 
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                native_contacts.append((i, j)) 
                native_contacts_atoms.append((atom_indices[i] + 1, atom_indices[j] + 1))
                r0_values.append(dist)
    return np.array(native_contacts, dtype=int).reshape(-1, 2), np.array(native_contacts_atoms, dtype=int).reshape(-1, 2), np.array(r0_values)

def q_switching(rij, r0, lambda_param, beta):
    z = beta * (rij - lambda_param * r0)
    # stable sigmoid avoiding overflow completely
    s = np.zeros_like(z)
    pos = z >= 0
    neg = ~pos
    # large positive z -> s ~ 0
    s[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    # large negative z -> s ~ 1
    s[neg] = 1.0 / (1.0 + np.exp(z[neg]))
    return s

def main():
    args = parse_arguments()

    # Load Universe
    u = mda.Universe(args.pdb, args.xtc)
    n_residues = len(u.select_atoms('name CA').resids)

    # reference frame
    ref = mda.Universe(args.pdb)

    # obtain sequence to check for presence of His-tags
    seq = ''.join(convert_aa_code(res.resname) for res in u.residues)
    pattern = "HHHHHH"
    if pattern in seq:
        tag_end = seq.find(pattern) + len(pattern) + 1 # convert to 1-indexed residues
        tag_end += 3 # add 3 more residues as His-tag is normally followed by "SSG"

        heavy_atoms = u.select_atoms(f"not name H* and resid {tag_end+1}:{n_residues}")
        heavy_atoms_ref = ref.select_atoms(f"not name H* and resid {tag_end+1}:{n_residues}")

    else:
        heavy_atoms = u.select_atoms("not name H*")
        heavy_atoms_ref = ref.select_atoms("not name H*")

    # get DSSP of reference structure
    traj = md.load(args.pdb)
    dssp = md.compute_dssp(traj, simplified=True)[0] # extract first reference frame
    structured_residues = np.where(dssp!='C')[0] + 1 # +1 to convert from 0-indxed to residue number starting from 1

    # Identify native contacts (sequence separation >=4 residues)
    native_contacts, native_contacts_atoms, r0_values = identify_native_contacts(heavy_atoms_ref, args.cutoff, structured_residues = structured_residues, min_seq_sep=4)

    if len(native_contacts) == 0:
        raise ValueError("No native contacts found with the given cutoff and sequence separation!")

    # Compute fraction of native contacts for each frame
    fractions = []
    for ts in u.trajectory:
        frame_positions = heavy_atoms.positions / 10.0  # Å -> nm
        rij = np.linalg.norm(frame_positions[native_contacts[:, 0]] - 
                             frame_positions[native_contacts[:, 1]], axis=1)
        sij = q_switching(rij, r0_values, args.lambda_param, args.beta)
        fractions.append(np.mean(sij))

    fractions = np.array(fractions)

    # Save to text file
    np.savetxt(args.output, fractions, fmt="%.6f")
    np.savetxt(args.output.split('/')[0]+str('/contacts.txt'), native_contacts_atoms, fmt="%.0f")
    print(f"Fraction of native contacts saved to {args.output}")

if __name__ == "__main__":
    main()
