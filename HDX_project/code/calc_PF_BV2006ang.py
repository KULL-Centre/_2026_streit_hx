import mdtraj as md
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--traj", type=str, help="Path to the trajectory file.")
parser.add_argument("--beta_c", type=float, default=0.35, help="Beta_c parameter. Default: 0.35")
parser.add_argument("--beta_h", type=float, default=2.00, help="Beta_h parameter. Default: 2.00")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--stride", type=int, default=1, help="Stride for analysing trajectory frames. Default: 1.")
parser.add_argument("--save_frames", type=str, default="False", help="Whether to save data for all frames in a numpy array. Default: False")

args = parser.parse_args()

pdb = args.pdb
traj = args.traj
beta_c = args.beta_c
beta_h = args.beta_h
save_dir = args.save_dir
stride = args.stride
save_framewise = args.save_frames

def compute_BVpf(trj, threshold_contact = 0.5, threshold_hbond = 0.24, treshold_angle = 150.0, beta_c = 0.35,  beta_h = 2.0, return_partial=False):
    """
    Compute protection factors according to Best-Vendruscolo 2006 model.
    This version also includes a hydrogen bond angle threshold to count hbonds (default 150 degrees or more like GROMACS hbond).
    We assume the mdtraj trajectory to contain the protein in chain 0 and no solvent. If there are multiple chains
    we assume them to be identical (i.e., dimers). The script will not work correctly if these assumptions are not met!
    Protection factors are computed only for residues containing an amide HN.
    Note that the returned BV protection factor is the natural log, ln(PF). Average first, then take np.exp(ln(PF))
    """
    
    threshold_contact = threshold_contact
    threshold_hbond = threshold_hbond
    beta_c = beta_c
    beta_h = beta_h
    
    
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices and residue numbers
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

    CAall = trj.top.select('name CA or name CH3 or (name N and resname NH2)') # CH3 in case of ACE and NH2 nitrogen atom in C-terminal NH2 resid
    all_resid = [trj.top.atom(i).residue.resSeq for i in CAall]
    n_residues = len(CAall)

    # check how many protein chains and if the chains are identical
    nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains
    
    contacts_res = np.zeros((len(trj), len(amide_res_idx)))
    hbonds_res = np.zeros((len(trj), len(amide_res_idx)))
    
    if nprot_chains==1:

        for i, resid in enumerate(amide_resid):
       
            remove_closest = f" and (resSeq < {max(np.min(all_resid),resid-2)} or resSeq > {min(np.max(all_resid),resid+2)})"
            # additional hydrogen bond acceptors other than O
            other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
            
            amideN_idx = trj.top.select(f"protein and name N and resSeq {resid}")
            heavy_idx = trj.top.select("protein and symbol != H" + remove_closest)
            ## NB: compute_neighbors is much faster than compute_distances
            neigh = md.compute_neighbors(trj, threshold_contact, amideN_idx, heavy_idx)
            contacts_res[:, i] = list(map(lambda x: len(x), neigh))
            
            
            amideH_idx = trj.top.select(f"protein and (name H or name HN) and resSeq {resid}")
            acceptor_idx = trj.top.select("protein and symbol O" + other_hbonds)
            neigh = md.compute_neighbors(trj, threshold_hbond, amideH_idx, acceptor_idx)
            
            # check D-H...A angles        
            hbonds_count = []
            for frame, frame_neighbors in enumerate(neigh):
                count = 0
                frame_xyz = trj.xyz[frame]  # only load coordinates for this frame

                for h_idx, donor_idx in zip(amideH_idx, amideN_idx):
                    vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                    vec_dh = vec_dh / np.linalg.norm(vec_dh)
            
                    for a_idx in frame_neighbors:
                        vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                        vec_ha = vec_ha / np.linalg.norm(vec_ha)
                        
                        angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                        if angle_rad >= np.deg2rad(treshold_angle):
                            count += 1
                hbonds_count.append(count)
            hbonds_res[:, i] = hbonds_count

    elif nprot_chains==2: # if two chains
        
        for i, resid in enumerate(amide_resid):

            # determine if it is the first or second chain
            if i < int(len(amide_res_idx)/2): # first chain

                #remove_closest = f" and not (chainid 0 and (resSeq > {max(1,resid-3)} and resSeq < {min(int(n_residues/2),resid+3)}))"
                remove_closest = f" and not (chainid 0 and resSeq {max(np.min(all_resid),resid-2)} to {min(np.max(all_resid),resid+2)})"
                # additional hydrogen bond acceptors other than O
                other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
                
                amideN_idx = np.array([trj.top.select(f"name N and resSeq {resid}")[0]]) # first match, first chain
                heavy_idx = trj.top.select("protein and symbol != H" + remove_closest)
                ## NB: compute_neighbors is much faster than compute_distances
                neigh = md.compute_neighbors(trj, threshold_contact, amideN_idx, heavy_idx)
                contacts_res[:, i] = list(map(lambda x: len(x), neigh))
                
                
                amideH_idx = np.array([trj.top.select(f"(name H or name HN) and resSeq {resid}")[0]]) # first match, first chain
                acceptor_idx = trj.top.select("protein and symbol O" + other_hbonds)
                neigh = md.compute_neighbors(trj, threshold_hbond, amideH_idx, acceptor_idx)
                hbonds_res[:, i] = list(map(lambda x: len(x), neigh))

                # check D-H...A angles        
                hbonds_count = []
                for frame, frame_neighbors in enumerate(neigh):
                    count = 0
                    frame_xyz = trj.xyz[frame]  # only load coordinates for this frame

                    for h_idx, donor_idx in zip(amideH_idx, amideN_idx):
                        vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                        vec_dh = vec_dh / np.linalg.norm(vec_dh)
                
                        for a_idx in frame_neighbors:
                            vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                            vec_ha = vec_ha / np.linalg.norm(vec_ha)
                            
                            angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                            if angle_rad >= np.deg2rad(treshold_angle):
                                count += 1
                    hbonds_count.append(count)
                hbonds_res[:, i] = hbonds_count

            else: # second chain
                remove_closest = f" and not (chainid 1 and resSeq {max(np.min(all_resid),resid-2)} to {min(np.max(all_resid),resid+2)})"
                # additional hydrogen bond acceptors other than O
                other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
                
                amideN_idx = np.array([trj.top.select(f"name N and resSeq {resid}")[1]]) # second match, second chain
                heavy_idx = trj.top.select("protein and symbol != H" + remove_closest)
                ## NB: compute_neighbors is much faster than compute_distances
                neigh = md.compute_neighbors(trj, threshold_contact, amideN_idx, heavy_idx)
                contacts_res[:, i] = list(map(lambda x: len(x), neigh))
                
                
                amideH_idx = np.array([trj.top.select(f"(name H or name HN) and resSeq {resid}")[1]]) # second match, second chain
                acceptor_idx = trj.top.select("protein and symbol O" + other_hbonds)
                neigh = md.compute_neighbors(trj, threshold_hbond, amideH_idx, acceptor_idx)
                hbonds_res[:, i] = list(map(lambda x: len(x), neigh))

                # check D-H...A angles        
                hbonds_count = []
                for frame, frame_neighbors in enumerate(neigh):
                    count = 0
                    frame_xyz = trj.xyz[frame]  # only load coordinates for this frame

                    for h_idx, donor_idx in zip(amideH_idx, amideN_idx):
                        vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                        vec_dh = vec_dh / np.linalg.norm(vec_dh)
                
                        for a_idx in frame_neighbors:
                            vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                            vec_ha = vec_ha / np.linalg.norm(vec_ha)
                            
                            angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                            if angle_rad >= np.deg2rad(treshold_angle):
                                count += 1
                    hbonds_count.append(count)
                hbonds_res[:, i] = hbonds_count


    else:
        print('Error: more than two chains are not supported')

    
    BestVendruscolo = beta_c * contacts_res + beta_h * hbonds_res
    if return_partial:
        return amide_res_idx, amide_resid, BestVendruscolo, contacts_res, hbonds_res
    else:
        return amide_res_idx, amide_resid, BestVendruscolo


# running
trj = md.load(traj,top=pdb)
if stride>1:
    trj = trj[::stride]
amide_res_idx, amide_resid, lnPF, contacts_res, hbonds_res = compute_BVpf(trj,beta_c=beta_c,beta_h=beta_h, return_partial=True)

if save_framewise=='True':
    np.save(save_dir+'contacts_res_frames_{}.npy'.format(traj[:-4]),contacts_res)
    np.save(save_dir+'hbonds_res_frames_{}.npy'.format(traj[:-4]),hbonds_res)

# ensemble average
avg_contacts_res = np.mean(contacts_res,axis=0)
avg_hbonds_res = np.mean(hbonds_res,axis=0)

# save data
np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
np.savetxt(save_dir+'amideresidues.txt',amide_resid)
np.savetxt(save_dir+'contacts_res_{}.txt'.format(traj[:-4]),avg_contacts_res)
np.savetxt(save_dir+'hbonds_res_{}.txt'.format(traj[:-4]),avg_hbonds_res)