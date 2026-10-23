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
def calc_M9(trj, sasa_threshold = 4.0, distance_threshold = 3.0):

    """
    Compute protection factors according model M9 by Mohammadiarani et al 2018.
    https://doi.org/10.1021/acs.jpcb.8b07494
    Protection factors are computed only for residues containing an amide HN.
    Note that the returned protection factor is the natural log, ln(PF). Average first, then take np.exp(ln(PF))
    Parameters have changed since original M8/M9 models used VMD to calculate SASA and the results differ
    the SASA values calculated with mdtraj are very similar to gmx sasa -ndots 960 -probe 0.14
    SASA and distance threshold are in Å^2 and Å, respectively.
    We assume the mdtraj trajectory to contain the protein in chain 0 and no solvent. If there are multiple chains
    we assume them to be identical (i.e., dimers). The script will not work correctly if these assumptions are not met!
    """

    
    protein_idxs = trj.top.select('protein or resname NH2 or resname ACE or resname NME')

    # calculate SASA for all frames - considering only protein atoms (in case ions or ligands are in the trajectory file)
    subtraj = trj.atom_slice(protein_idxs)
    amide_idxs = subtraj.top.select('name H or name HN')
    sasa = md.shrake_rupley(subtraj, probe_radius=0.14, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
    amideH_SASA = sasa[:,amide_idxs]

    # calculate amide H distance to closest polar atom (N, O, F, S)
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

    # make sure number of residues with H/HN atoms matches the initial selection
    assert len(amide_resid)==amide_idxs.shape[0]

    # check how many protein chains and if the chains are identical
    nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains

    # all residues
    CAall = trj.top.select('name CA')
    n_residues = len(CAall)

    min_distances = []

    if nprot_chains==1:
    
        for i, resid in enumerate(amide_resid):

            remove_closest = f" and (resSeq < {max(np.min(amide_resid),resid-1)} or resSeq > {min(np.max(amide_resid),resid+1)})"
            amideH_idx = trj.top.select(f"(name H or name HN) and resSeq {resid}")
            polar_idx = trj.top.select("protein and (symbol O or symbol N or symbol S or symbol F)" + remove_closest)
            atom_pairs = []
            for idx in polar_idx:
                atom_pairs.append([amideH_idx[0],idx])
            atom_pairs = np.array(atom_pairs)
            distances = md.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
            min_distances.append(np.min(distances, axis=1))

    elif nprot_chains==2: # if two chains

        for i, resid in enumerate(amide_resid):

            # determine if it is the first or second chain
            if i < int(len(amide_res_idx)/2): # first chain

                remove_closest = f" and not (chainid 0 and resSeq {max(np.min(amide_resid),resid-1)} to {min(np.max(amide_resid),resid+1)})"
                amideH_idx = np.array([trj.top.select(f"(name H or name HN) and resSeq {resid}")[0]]) # first match, first chain
                polar_idx = trj.top.select("protein and (symbol O or symbol N or symbol S or symbol F)" + remove_closest)
                atom_pairs = []
                for idx in polar_idx:
                    atom_pairs.append([amideH_idx[0],idx])
                atom_pairs = np.array(atom_pairs)
                distances = md.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
                min_distances.append(np.min(distances, axis=1))

            else: # second chain

                remove_closest = f" and not (chainid 1 and resSeq {max(np.min(amide_resid),resid-1)} to {min(np.max(amide_resid),resid+1)})"
                amideH_idx = np.array([trj.top.select(f"(name H or name HN) and resSeq {resid}")[1]]) # second match, second chain
                polar_idx = trj.top.select("protein and (symbol O or symbol N or symbol S or symbol F)" + remove_closest)
                atom_pairs = []
                for idx in polar_idx:
                    atom_pairs.append([amideH_idx[0],idx])
                atom_pairs = np.array(atom_pairs)
                distances = md.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
                min_distances.append(np.min(distances, axis=1))

    else:
        print('Error: more than two chains are not supported')


    min_distances = np.array(min_distances)
    min_distances = min_distances.T

    # calculate protection factors M9
    # open state corresponds to amides exceding a sasa_threshold and having no polar protein atom within the distance threshold
    
    bool_SASA = (amideH_SASA > sasa_threshold)*1
    bool_dist = (min_distances > distance_threshold)*1
    bool_sum = bool_SASA + bool_dist
    bool_open = (bool_sum==2)*1 # both conditions fulfilled

    return amide_resid, amide_res_idx, bool_open

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
            amide_resid, amide_res_idx, fopen = calc_M9(trj)

        else:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, tmpfopen = calc_M9(trj)
            # append
            fopen = np.concatenate((fopen, tmpfopen),axis=0)

    # save frame-wise data
    np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
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


