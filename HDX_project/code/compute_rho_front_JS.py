import argparse
from pathlib import Path
import mdtraj as md
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--traj", required=True)
parser.add_argument("--save_dir", default="outputs")
parser.add_argument("--stride", type=int, default=1)

# distance scale for switching function, in nm
parser.add_argument("--r_cut", type=float, default=0.40)
parser.add_argument("--cutoff_nm", type=float, default=0.50) # for neighbour search, in nm

# angular weighting: max(0, cos(theta)) ** angular_power
parser.add_argument("--angular_power", type=float, default=1.0)

args = parser.parse_args()

Path(args.save_dir).mkdir(parents=True, exist_ok=True)

top = md.load(args.pdb).topology


# function to calculate open/closed status
def compute_rho(trj, distance_cutoff = 0.5, angular_power = 1.0, switch_exp = 12, switch_r0 = 0.4, remove_closest = True):

    """
    Compute the rho front for each amide residue in the trajectory.
    """

    # calculate hbond contacts within threshold and their angles
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices and residue numbers
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]
    amide_chain_idx = [trj.top.atom(i).residue.chain.index for i in CAind]

    CAall = trj.top.select('name CA or name CH3 or (name N and resname NH2)') # CH3 in case of ACE and NH2 nitrogen atom in C-terminal NH2 resid
    all_resid = [trj.top.atom(i).residue.resSeq for i in CAall]
    n_residues = len(CAall)

    # check how many protein chains and if the chains are identical
    nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains
    
    rho_res = np.zeros((len(trj), len(amide_res_idx)))
    
    for i, resid in enumerate(amide_res_idx):
        amideH_idx = trj.top.select(f"protein and (name H or name HN) and resid {resid}")
        amideN_idx = trj.top.select(f"protein and name N and resid {resid}")
        if remove_closest:
            acceptor_idx = trj.top.select(f"(not resid {max(resid-1,0)} to {resid}) and protein and not symbol H") # heavy atoms
        else:
            acceptor_idx = trj.top.select("protein and not symbol H") # heavy atoms
        neigh = md.compute_neighbors(trj, distance_cutoff, amideH_idx, acceptor_idx)
        # check D-H...A angles        
        rho_values = []
        for frame, frame_neighbors in enumerate(neigh):
            frame_xyz = trj.xyz[frame]  # only load coordinates for this frame
            terms = []
            for h_idx, donor_idx in zip(amideH_idx, amideN_idx):
                vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                vec_dh = vec_dh / np.linalg.norm(vec_dh)
                for a_idx in frame_neighbors:
                    vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                    vec_ha = vec_ha / np.linalg.norm(vec_ha)
                    angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                    dist = np.linalg.norm(frame_xyz[h_idx] - frame_xyz[a_idx])
                    switch = 1 / (1 + (dist/switch_r0)**switch_exp)
                    angle_weight = max(0, np.cos(angle_rad)) ** angular_power
                    rho = switch * angle_weight
                    terms.append(rho)
            rho_values.append(sum(terms))   
        rho_res[:, i] = rho_values

    return amide_resid, amide_res_idx, amide_chain_idx, rho_res


# running
trj = md.load(args.traj,top=args.pdb)
if args.stride>1:
    trj = trj[::args.stride]
amide_resid, amide_res_idx, amide_chain_idx, rho_res = compute_rho(trj, distance_cutoff=args.cutoff_nm, angular_power=args.angular_power, switch_r0=args.r_cut)

# save data
np.save(args.save_dir+'rho_res_frames_{}.npy'.format(args.traj[:-4]),rho_res)
np.savetxt(args.save_dir+'amideresidx.txt',amide_res_idx)
np.savetxt(args.save_dir+'amideresidues.txt',amide_resid)