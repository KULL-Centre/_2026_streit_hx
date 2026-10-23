import mdtraj as mdj
import numpy as np
import sys

pdb = str(sys.argv[1])
traj = str(sys.argv[2])
save_dir = str(sys.argv[3])

def calc_SASA(trj, probe_radius=0.14):

    """
    Compute AMIDE SASA.
    https://doi.org/10.1021/acs.jpcb.8b07494
    Provided probe radius is in nm, but the output will be in Angstrom^2 (1 nm^2 = 100 Angstrom^2).
    """

    
    protein_idxs = trj.top.select('protein or resname NH2 or resname ACE or resname NME')

    # calculate SASA for all frames - considering only protein atoms (in case ions or ligands are in the trajectory file)
    subtraj = trj.atom_slice(protein_idxs)
    amide_idxs = subtraj.top.select('name H or name HN')
    sasa = mdj.shrake_rupley(subtraj, probe_radius=probe_radius, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
    amideH_SASA = sasa[:,amide_idxs]

    CAind = trj.top.select('(name H or name HN) and not (resname PRO or resname NH2)')
    # Get the residue indices
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

    return amide_res_idx, amide_resid, amideH_SASA


# running
trj = mdj.load(traj,top=pdb)
amide_res_idx, amide_resid, amideH_SASA = calc_SASA(trj, probe_radius=0.14)

# save data
np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
np.savetxt(save_dir+'amideresidues.txt',amide_resid)
np.save(save_dir+'amideH_SASA_{}.npy'.format(traj[:-4]),amideH_SASA)

