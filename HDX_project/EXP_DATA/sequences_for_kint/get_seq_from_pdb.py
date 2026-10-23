## WARNING: ONLY WORKS ON SINGLE CHAIN PDB FILES - REQUIRES PROCESSING OF PDB FILES FIRST CHAIN BY CHAIN

import mdtraj as mdj
import numpy as np
import sys

def seq_to_list(seq):
    if 'Pc' not in seq and 'C2' not in seq:
        return list(seq)
    else:
        interm = list(seq)
        final = []
        for ele in interm:
            if ele != 'c' and ele != '2':
                final.append(ele)
            else:
                final[-1]=final[-1]+ele

        return list(final)

pdb = sys.argv[1]

u = mdj.load(pdb,top=pdb)
residues = list(u.topology.residues)
n_residues = u.topology.n_residues
initial_sequence = u.topology.to_fasta()[0]

# add non-standard termini to sequence
if 'ACE' in str(residues[0]):
    initial_sequence = str('X') + initial_sequence
if 'NME' in str(residues[-1]):
    initial_sequence+= 'X'
if 'NH2' in str(residues[-1]):
    initial_sequence+= 'X'

if len(initial_sequence)!=n_residues:
    print(pdb)

assert len(initial_sequence)==n_residues

initial_seq_list = seq_to_list(initial_sequence)

# find cis prolines and update sequence
atoms, omegas = mdj.compute_omega(u)
omegas = np.abs(omegas[0])
cis_idxs = np.where(omegas<np.deg2rad(50))[0] # cis peptide bonds defined with abs(omega) < 50 degrees

if cis_idxs.shape[0]>0:

    cis_residues = []
    for idx in cis_idxs:
        atom = atoms[idx][-1]
        residue_idx = u.top.atom(atom).residue.index
        if u.top.atom(atom).residue.name =='PRO':
            # replace P with Pc
            initial_seq_list[residue_idx] = str('Pc')
            
else:
    initial_seq_list = initial_seq_list




# find disulfides and update sequence
CA_Cys_idxs = u.top.select('resname CYS and name CA')
for idx in CA_Cys_idxs:
    residue_idx = u.top.atom(idx).residue.index
    cys = u.top.select('resid {} and resname CYS'.format(residue_idx))
    cysatoms = [u.top.atom(i).name for i in cys]

    if 'HG' not in cysatoms:
        initial_seq_list[residue_idx] = str('C2') 
    else:
        initial_seq_list = initial_seq_list

# final sequence
sequence = ''.join(initial_seq_list)

# check that final sequence is the correct length
assert len(seq_to_list(sequence))==n_residues

# write sequence
filename = pdb[:-4]+str('.txt')
with open(filename,'w') as f:
    f.write(sequence)



