import mdtraj as md
import pandas as pd
import numpy as np
import argparse
from rew_functions import *
from utils import *

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--min_neff", type=float, default=0.1, help="Minimum threshold for the fraction of effective frames. Default: 0.1")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")
parser.add_argument("--w0", type=str, default=None, help="Optional filename with initial weights for reweighting. If not provided, uniform weights will be used.")
parser.add_argument('--rew_method', type=str, default='linear', 
            help='Reweighting method: linear (fraction open space using BME/Lagrange multipliers), log (ln PF space directly over weights), log_lambda (ln PF space over lambda parameters)')
parser.add_argument("--filter", action="store_true", help="Filter out frames with overlapping atoms (e.g. for BioEmu - not necessary for all-atom MD ensembles).")

args = parser.parse_args()

pdb = args.pdb
protname = args.name
save_dir = args.save_dir
Neff_threshold = args.min_neff
max_frames_per_traj = args.max_frames
w0 = np.loadtxt(args.w0) if args.w0 else None
rew_method = args.rew_method
filter = args.filter

if rew_method=='log':
    save_prefix = 'log_'
elif rew_method=='log_lambda':
    save_prefix = 'log_lambda_'
elif rew_method=='linear':
    save_prefix = ''
else:
    print('Error: did not recognise reweighting method: ', rew_method)

# function to calculate open/closed status
def calc_M13(trj, sasa_threshold = 0.5, distance_threshold = 0.26, angle_threshold = 120.0, remove_closest = True):

    """
    Compute protection factors (model 10) according to hbond status and SASA.
    Closed state: hbond distance < 0.26 nm & angle < 120 deg OR SASA < 0.5 Angstrom^2. 
    This version counts all hydrogen bonds excluding within the same / preceding residue (if remove_closest=True)
    as long as the distance and angle thresholds are satisfied.
    Open status (0=closed, 1=open) are computed only for residues containing an amide HN.
    We assume the mdtraj trajectory to contain the protein atoms and no solvent. Multiple chains are handled.
    """

    protein_idxs = trj.top.select('protein or resname NH2 or resname ACE or resname NME')

    # calculate SASA for all frames - considering only protein atoms (in case ions or ligands are in the trajectory file)
    subtraj = trj.atom_slice(protein_idxs)
    amide_idxs = subtraj.top.select('protein and (name H or name HN)')
    sasa = md.shrake_rupley(subtraj, probe_radius=0.14, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
    amideH_SASA = sasa[:,amide_idxs]

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
    
    hbonds_res = np.zeros((len(trj), len(amide_res_idx)))
    
    for i, resid in enumerate(amide_res_idx):
        # hydrogen bond acceptors other than O
        other_hbonds = f" or protein and symbol F or (protein and symbol N and (resid != {resid} or name != N))"
        amideH_idx = trj.top.select(f"protein and (name H or name HN) and resid {resid}")
        amideN_idx = trj.top.select(f"protein and name N and resid {resid}")
        if remove_closest:
            acceptor_idx = trj.top.select(f"(not resid {max(resid-1,0)} to {resid}) and protein and symbol O" + other_hbonds)
        else:
            acceptor_idx = trj.top.select("protein and symbol O" + other_hbonds)
        neigh = md.compute_neighbors(trj, distance_threshold, amideH_idx, acceptor_idx)
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
                    if angle_rad >= np.deg2rad(angle_threshold):
                        count += 1
            hbonds_count.append(count)
        hbonds_res[:, i] = hbonds_count

    # calculate protection factors M10
    # open state corresponds to amides exceding a sasa_threshold and having no polar protein atom within the distance threshold
    
    bool_SASA = (amideH_SASA > sasa_threshold)*1    # open with SASA > 0 Å^2
    bool_hbond = (hbonds_res == 0)*1                # open with no h-bond (distance & angle)
    bool_sum = bool_SASA + bool_hbond
    bool_open = (bool_sum==2)*1                     # both conditions fulfilled = open (positive SASA & no hbond)

    return amide_resid, amide_res_idx, amide_chain_idx, bool_open

### PERFORM FORWARD MODEL CALCULATIONS ###

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 

# check if numpy files already exist (i.e., forward model calculations done)
npyfiles = sorted(find_npy_filenames(save_dir))

if len(npyfiles)==0:

    for traj in trajs:
        trj = md.load(traj,top=pdb)
        if not filter:
            trj = trj
        else:
            ok_frames = filter_frames(pdb, traj)
            trj = trj[ok_frames]
            np.savetxt(save_dir+f'okframes_{traj[:-4]}.txt', ok_frames, fmt='%d')
        
        nframes = len(trj)

        # if trajectory contains more frames than set by the max threshold (due to memory considerations for reweighting) - subsample the traj
        if nframes > max_frames_per_traj:

            subsampleN = int(np.round(nframes/max_frames_per_traj, 0))
            trj = trj[::subsampleN]
        
        print('calculating PF for {}'.format(traj))
        if traj==trajs[0]:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, amide_chain_idx, fopen = calc_M13(trj)

        else:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, amide_chain_idx, tmpfopen = calc_M13(trj)
            # append
            fopen = np.concatenate((fopen, tmpfopen),axis=0)

    # save frame-wise data
    np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
    np.savetxt(save_dir+'amidechainidx.txt',amide_chain_idx)
    np.save(save_dir+'fopen.npy',fopen)

    # prepare sim data
    amide_resid = np.loadtxt(save_dir+'amideresidues.txt')
    fopen = np.load(save_dir+'fopen.npy').astype(float)

    # prepare sim data
    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(fopen)

    # load experimental data
    exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

    # reweight
    print('Reweighting...')
    if w0 is not None:
        print('Using provided initial weights for reweighting (--w0)')
    else:
        print('Using uniform prior weights, no alternative prior weights were provided (--w0): w0=', w0)

    
    if rew_method=='linear':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    elif rew_method=='log':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_softmax(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    elif rew_method=='log_lambda':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_lambda(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:
        print('Error: did not recognise reweighting method: ', rew_method)
    
    # elbow, choose the point just after it
    el_idx, el_x, el_y = find_elbow(reweighting_data[1, :], reweighting_data[0, :])
    el_idx += 1
    wopt_elbow = wopt_array[el_idx]

    # save data
    stats_df.to_csv(save_dir+f'{save_prefix}stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+f'{save_prefix}wopt_array.npy',wopt_array)
    np.save(save_dir+f'{save_prefix}wopt.npy',wopt)
    np.save(save_dir+f'{save_prefix}wopt_elbow.npy',wopt_elbow)
    np.savetxt(save_dir+f'{save_prefix}rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+f'{save_prefix}reweighting_chi2_phi_theta.txt',reweighting_data)

else:

    # prepare sim data
    amide_resid = np.loadtxt(save_dir+'amideresidues.txt')
    fopen = np.load(save_dir+'fopen.npy').astype(float)

    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(fopen)

    # load experimental data
    exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

    # reweight
    print('Reweighting...')
    if w0 is not None:
        print('Using provided initial weights for reweighting (--w0)')
    else:
        print('Using uniform prior weights, no alternative prior weights were provided (--w0): w0=', w0)

    
    if rew_method=='linear':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    elif rew_method=='log':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_softmax(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    elif rew_method=='log_lambda':
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_lambda(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:
        print('Error: did not recognise reweighting method: ', rew_method)
    
    # elbow, choose the point just after it
    el_idx, el_x, el_y = find_elbow(reweighting_data[1, :], reweighting_data[0, :])
    el_idx += 1
    wopt_elbow = wopt_array[el_idx]

    # save data
    stats_df.to_csv(save_dir+f'{save_prefix}stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+f'{save_prefix}wopt_array.npy',wopt_array)
    np.save(save_dir+f'{save_prefix}wopt.npy',wopt)
    np.save(save_dir+f'{save_prefix}wopt_elbow.npy',wopt_elbow)
    np.savetxt(save_dir+f'{save_prefix}rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+f'{save_prefix}reweighting_chi2_phi_theta.txt',reweighting_data)


