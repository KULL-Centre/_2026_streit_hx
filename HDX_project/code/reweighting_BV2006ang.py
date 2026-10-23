import mdtraj as md
import pandas as pd
import numpy as np
import argparse
from rew_functions import *
from utils import *

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--beta_c", type=float, default=0.89, help="Beta_c parameter. Default: 0.89")
parser.add_argument("--beta_h", type=float, default=4.39, help="Beta_h parameter. Default: 4.39")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--min_neff", type=float, default=0.1, help="Minimum threshold for the fraction of effective frames. Default: 0.1")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")
parser.add_argument("--w0", type=str, default=None, help="Optional filename with initial weights for reweighting. If not provided, uniform weights will be used.")

args = parser.parse_args()

pdb = args.pdb
protname = args.name
beta_c = args.beta_c
beta_h = args.beta_h
save_dir = args.save_dir
Neff_threshold = args.min_neff
max_frames_per_traj = args.max_frames
w0 = np.loadtxt(args.w0) if args.w0 else None

def compute_BVpf(trj, threshold_contact = 0.5, threshold_hbond = 0.24, treshold_angle = 150.0, beta_c = beta_c,  beta_h = beta_h, return_partial=False):
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
        return amide_resid, amide_res_idx, BestVendruscolo, contacts_res, hbonds_res
    else:
        return amide_resid, amide_res_idx, BestVendruscolo

### PERFORM FORWARD MODEL CALCULATIONS ###

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 

# check if numpy files already exist (i.e., forward model calculations done)
npyfiles = sorted(find_npy_filenames(save_dir))

if len(npyfiles)==0:

    for traj in trajs:
        trj = md.load(traj,top=pdb)

        nframes = len(trj)

        # if trajectory contains more frames than set by the max threshold (due to memory considerations for reweighting) - subsample the traj
        if nframes > max_frames_per_traj:
        
            subsampleN = int(np.round(nframes/max_frames_per_traj, 0))
            trj = trj[::subsampleN]
        
        print('calculating PF for {}'.format(traj))
        if traj==trajs[0]:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, lnPF, contacts_res, hbonds_res = compute_BVpf(trj, beta_c = beta_c, beta_h = beta_h, return_partial=True)

        else:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, tmp_lnPF, tmp_contacts_res, tmp_hbonds_res = compute_BVpf(trj, beta_c = beta_c, beta_h = beta_h, return_partial=True)
            # append
            contacts_res = np.concatenate((contacts_res, tmp_contacts_res), axis=0)
            hbonds_res = np.concatenate((hbonds_res, tmp_hbonds_res), axis=0)

    # save frame-wise data
    np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
    np.save(save_dir+'contacts_res_frames.npy',contacts_res)
    np.save(save_dir+'hbonds_res_frames.npy',hbonds_res)

    # prepare sim data
    amide_resid = np.loadtxt(save_dir+'amideresidues.txt')
    contacts_res = np.load(save_dir+'contacts_res_frames.npy')
    hbonds_res = np.load(save_dir+'hbonds_res_frames.npy')
    lnPF = contacts_res * beta_c + hbonds_res * beta_h

    # prepare sim data
    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(lnPF)

    # load experimental data
    exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

    # reweight
    print('Reweighting...')
    if w0 is not None:
        print('Using provided initial weights for reweighting')
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_lnPF_BV(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:
        w0 = None
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_lnPF_BV(exp, sim_data, Neff_threshold=Neff_threshold)

    # elbow, choose the point just after it
    el_idx, el_x, el_y = find_elbow(reweighting_data[1, :], reweighting_data[0, :])
    el_idx += 1
    wopt_elbow = wopt_array[el_idx]

    # save data
    stats_df.to_csv(save_dir+'stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+'wopt_array.npy',wopt_array)
    np.save(save_dir+'wopt.npy',wopt)
    np.save(save_dir+'wopt_elbow.npy',wopt_elbow)
    np.savetxt(save_dir+'rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+'reweighting_chi2_phi_theta.txt',reweighting_data)
    
else:

    # prepare sim data
    amide_resid = np.loadtxt(save_dir+'amideresidues.txt')
    contacts_res = np.load(save_dir+'contacts_res_frames.npy')
    hbonds_res = np.load(save_dir+'hbonds_res_frames.npy')
    lnPF = contacts_res * beta_c + hbonds_res * beta_h

    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(lnPF)

    # load experimental data
    exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

    # reweight
    print('Reweighting...')
    if w0 is not None:
        print('Using provided initial weights for reweighting')
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_lnPF_BV(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:
        w0 = None
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_lnPF_BV(exp, sim_data, Neff_threshold=Neff_threshold)

    # elbow, choose the point just after it
    el_idx, el_x, el_y = find_elbow(reweighting_data[1, :], reweighting_data[0, :])
    el_idx += 1
    wopt_elbow = wopt_array[el_idx]

    # save data
    stats_df.to_csv(save_dir+'stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+'wopt_array.npy',wopt_array)
    np.save(save_dir+'wopt.npy',wopt)
    np.save(save_dir+'wopt_elbow.npy',wopt_elbow)
    np.savetxt(save_dir+'rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+'reweighting_chi2_phi_theta.txt',reweighting_data)

