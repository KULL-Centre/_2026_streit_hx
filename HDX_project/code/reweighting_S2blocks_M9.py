import mdtraj as md
import MDAnalysis as mda
import pandas as pd
import numpy as np
import sys
import os
import scipy.stats
bme_dir = '/projects/lindorff_group/people/thd711/HDX_project/code/BME/'
sys.path.append(bme_dir)
import BME as BME
import argparse

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--min_neff", type=float, default=0.1, help="Minimum threshold for the fraction of effective frames. Default: 0.1")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")

args = parser.parse_args()

pdb = args.pdb
protname = args.name
save_dir = args.save_dir
Neff_threshold = args.min_neff
max_frames_per_traj = args.max_frames

# hard coded parameter for tolerance of deviation from target Neff
neff_diff_tol = 0.05

def find_xtc_filenames( path_to_dir, suffix=".xtc" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def find_npy_filenames( path_to_dir, suffix=".npy" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# functions to calculate statistics
def RMSE(actual, predicted):
    
    # Convert inputs to numpy arrays 
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the squared differences
    squared_diff = (actual - predicted) ** 2
    
    # Compute the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Take the square root of the mean squared difference
    return np.sqrt(mean_squared_diff)

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

def reweight(exp, sim_data, Neff_threshold=Neff_threshold):

    # simulated data
    sim_res = sim_data[0]
    sim_calc = sim_data[1] # fopen

    # filter out residues that have an average fopen = 0 (i.e. no open state sampled)
    mu_calc = np.mean(sim_calc,axis=0)
    valid = np.array(np.where(mu_calc>0)[0]).astype(int)
    sim_res = sim_res[valid]
    invalid = np.array(np.where(mu_calc==0)[0]).astype(int)
    invalid_residues = sim_data[0][invalid]
    sim_calc = sim_calc[:,valid]

    # common residues overlapping between calculated and experiment
    common_residues = np.intersect1d(sim_res, np.array(exp['Model_residue']))

    idxscalc = []
    for res in common_residues:
        idxscalc.append(list(sim_res).index(res))
    idxscalc = np.array(idxscalc)

    idxsexp = []
    for res in common_residues:
        idxsexp.append(list(exp['Model_residue']).index(res))
    idxsexp = np.array(idxsexp)

    # final arrays

    labels = np.array(exp['Model_residue']).astype(float) # data point labels = residue numbers
    exp_vals = 1 / np.array(exp['PF']).astype(float) # converted to fopen (1 / PF)
    exp_errors = np.array(exp['PF_error']).astype(float) / ( np.array(exp['PF']).astype(float) ** 2 ) # propagate error for 1 / PF (error = PF_error / PF**2)

    # sub index to match experimental to calculated data
    calc = sim_calc[:,idxscalc]
    exp_vals = exp_vals[idxsexp]
    exp_errors = exp_errors[idxsexp]
    labels = labels[idxsexp]

    n_frames = calc.shape[0]
    w0 = np.ones(n_frames) / n_frames  # uniform initial weights

    bounds = np.zeros(exp_vals.shape[0]).astype(int)
    EXP = np.vstack([exp_vals, exp_errors, bounds]).T

    # warning about residues not sampling open state but present in experimental dataset
    final_invalid_residues = []
    for res in invalid_residues:
        if res in list(exp['Model_residue']):
            final_invalid_residues.append(res)
    print('Warning: The following residues with experimental PFs have not sampled the open state: ', final_invalid_residues)

    # Create Reweight object with name and optional initial weights
    rew_obj = BME.Reweight("BME", w0=w0)

    # load array
    rew_obj.load_array(labels,EXP,calc)

    # Reweight using fit() 
    #thetas = np.geomspace(1e6,1e0,50)
    thetas = np.geomspace(1e8,1e0,50)
    wopt_array = []
    final_thetas = []
    final_phis = []
    final_chi2 = []
    for theta in thetas:
        chi2_before, chi2_after, phi = rew_obj.fit(theta=theta)

        if phi < Neff_threshold - neff_diff_tol:
            break

        if chi2_after>0.0:
            wopt_array.append(rew_obj.get_weights())
            final_thetas.append(theta)
            final_phis.append(phi)
            final_chi2.append(chi2_after)


    # If still too high phi, do a fine scan (small theta values) - fine scan 1
    if np.min(final_phis) > Neff_threshold + neff_diff_tol:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-2, 100)
        for theta in fine_thetas:
            chi2_before, chi2_after, phi = rew_obj.fit(theta=theta)
            if phi < Neff_threshold - neff_diff_tol:
                break
            if chi2_after > 0.0:
                wopt_array.append(rew_obj.get_weights())
                final_thetas.append(theta)
                final_phis.append(phi)
                final_chi2.append(chi2_after)

    # If still too high phi, do a fine scan (small theta values) - fine scan 2
    if np.min(final_phis) > Neff_threshold + neff_diff_tol:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-6, 200)
        for theta in fine_thetas:
            chi2_before, chi2_after, phi = rew_obj.fit(theta=theta)
            if phi < Neff_threshold - neff_diff_tol:
                break
            if chi2_after > 0.0:
                wopt_array.append(rew_obj.get_weights())
                final_thetas.append(theta)
                final_phis.append(phi)
                final_chi2.append(chi2_after)
            

    # Select best theta: max theta for which phi >= threshold - tol
    calc = sim_calc[:,idxscalc] # re-load calculated data before normalisation
    best_index = -1 # default
    for i, phi in enumerate(final_phis):
        curr_weighted_avg = np.average(calc,axis=0,weights=wopt_array[i]) # calculated weighted average of fopen array to make sure no residues are 0
        # pick the last index where weights result in fopen>0 (for all residues) and phi is above the threshold within tolerance
        if np.any(curr_weighted_avg == float(0))==False and phi >= Neff_threshold - neff_diff_tol: 
            best_index = i
    
    wopt=wopt_array[best_index]

    # only keep data until the "best_index"
    wopt_array = wopt_array[:best_index+1]
    final_thetas = final_thetas[:best_index+1]
    final_phis = final_phis[:best_index+1]
    final_chi2 = final_chi2[:best_index+1]

    # calculate statistics for prior
    calc = sim_calc[:,idxscalc]
    lnPF_rew= np.log(1 / np.average(calc,axis=0,weights=w0))
    x, y = np.log10(exp['PF'][idxsexp]), np.log10(np.exp(lnPF_rew))
    rpearson_prior = scipy.stats.pearsonr(x,y)[0]
    rmse_prior = RMSE(x,y)

    # calculate statistics for posterior
    lnPF_rew= np.log(1 / np.average(calc,axis=0,weights=wopt))
    x, y = np.log10(exp['PF'][idxsexp]), np.log10(np.exp(lnPF_rew))
    rpearson_post = scipy.stats.pearsonr(x,y)[0]
    rmse_post = RMSE(x,y)

    # summarise stats
    stats_df = pd.DataFrame()
    stats_df['rpearson prior'] = [rpearson_prior]
    stats_df['rmse prior'] = [rmse_prior]
    stats_df['rpearson post'] = [rpearson_post]
    stats_df['rmse post'] = [rmse_post]
    stats_df['final neff'] = [final_phis[best_index]]
    stats_df['Exp residues no open state'] = [final_invalid_residues] # experimental residues that in the model have not sampled the open state

    # calculate reweighted lnPF
    rew_lnPF = np.log(1 / np.average(calc, weights=wopt, axis=0))
    prior_lnPF = np.log(1 / np.average(calc, weights=w0, axis=0))

    return labels, stats_df, wopt_array, wopt, np.array([final_chi2, final_phis, final_thetas]), rew_lnPF, prior_lnPF

# functions for order parameters
def nh_bond_vectors(u, selN='name N and not resname PRO', selH='(name H or name HN) and not resname PRO'):
    """
    Compute unit N–H bond vectors for all residues across the trajectory.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Loaded trajectory.
    selN, selH : str
        Atom selection strings for N and H atoms.

    Returns
    -------
    nh_vectors : np.ndarray, shape (n_frames, n_bonds, 3)
        Unit N–H bond vectors per frame.
    resids : np.ndarray, shape (n_bonds,)
        Residue IDs corresponding to each bond.
    """
    # exclude first residue
    first_resid = u.select_atoms('name CA').resids[0]

    selN = u.select_atoms(selN + f" and not resid {first_resid}")
    selH = u.select_atoms(selH + f" and not resid {first_resid}")

    if len(selN) != len(selH):
        raise ValueError(f"Selections mismatch: {len(selN)} N vs {len(selH)} H atoms")

    n_frames = len(u.trajectory)
    n_bonds = len(selN)
    nh_vectors = np.zeros((n_frames, n_bonds, 3), dtype=np.float32)

    for iframe, ts in enumerate(u.trajectory):
        # difference: H - N (direction from N to H)
        vecs = selH.positions - selN.positions
        # normalize each bond vector
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        nh_vectors[iframe] = vecs

    return nh_vectors, selN.resids

def compute_order_parameters(nh_vectors, weights=None, scaling_factor=1.0):
    """
    Compute amide order parameters (S²) from N–H bond unit vectors.

    Parameters
    ----------
    nh_vectors : np.ndarray, shape (n_frames, n_bonds, 3)
        Unit N–H bond vectors over time.
    weights : np.ndarray, shape (n_frames,), optional
        Optional frame weights (e.g., from reweighting). If None, uniform weights are used.

    Returns
    -------
    S2 : np.ndarray, shape (n_bonds,)
        Order parameter per N–H bond.
    Q : np.ndarray, shape (n_bonds, 3, 3)
        Orientation tensors (⟨v_iα v_iβ⟩ for each bond)
    """
    n_frames, n_bonds, _ = nh_vectors.shape
    if weights is None:
        weights = np.ones(n_frames, dtype=np.float64) / n_frames
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= np.sum(weights)  # normalize

    # weighted mean of outer products for each bond
    # <v v^T> per bond
    Q = np.einsum('t, tij, tik -> ijk', weights, nh_vectors, nh_vectors)

    # compute S^2 = 1.5 * sum(Q^2) - 0.5
    S2 = 1.5 * np.sum(Q**2, axis=(1,2)) - 0.5

    # clip numerical noise
    S2 = np.clip(S2, 0.0, 1.0)

    return S2 * scaling_factor

### PERFORM FORWARD MODEL CALCULATIONS ###

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 
time_factors = [0.5, 1, 2, 3, 5, 10]

# find TAUR from info file
info = pd.read_csv(f'../../ORTHOGONAL_EXP_DATA/{protname}/info.csv')
TAUR = float(info[info['Experiment']=='S2_NH']['S2_iso_tauc'])

# parameters for time-averaging / MD data
frame_dt_ps = 100.0

for traj in trajs:
    ###
    # calculate order parameters using 0.5 x , 1 x , 2 x, 3 x, 5 x, 10 x the value of TAUR for time-averaging blocks
    ###
    trj = mda.Universe(pdb, traj)
    nframes = len(trj.trajectory)

    # if trajectory contains more frames than set by the max threshold (due to memory considerations for reweighting) - subsample the traj
    if nframes > max_frames_per_traj:

        subsampleN = int(np.round(nframes/max_frames_per_traj, 0))
        trj = trj[::subsampleN]

    print('calculating amide bond vectors/S2 for {}'.format(traj))
    if traj==trajs[0]:
        # run NH bond vector calculation
        vec, S2resids = nh_bond_vectors(trj)

    else:
        # run NH bond vector calculation
        tmpvec, S2resids = nh_bond_vectors(trj)
        # append
        vec = np.concatenate((vec, tmpvec),axis=0)

    # frames from NH bond vector data
    n_frames, n_bonds, _ = vec.shape


# compute order parameters using time-averaging of blocks
for time_factor in time_factors:
    print(f'Time averaging order parameters with window size {time_factor} x TAUR')

    window_ns = TAUR * time_factor
    # convert ns -> frames (frame_dt is in ps)
    frames_per_window = max(1, int(round(window_ns * 1000.0 / float(frame_dt_ps))))
    starts = list(range(0, n_frames - frames_per_window + 1, frames_per_window))

    S2_block_frames = []

    for iw, ws in enumerate(starts):
        we = ws + frames_per_window  # exclusive
        block = vec[ws:we]   # shape (fw, n_bonds, 3)
        S2 = compute_order_parameters(block)
        S2_block_frames.append(S2)

    S2_block_frames = np.array(S2_block_frames)
    # save S2 block data
    np.save(save_dir+f'S2_blocks_{time_factor}xTAUR.npy', S2_block_frames)
    np.savetxt(save_dir+'S2_residues.txt', S2resids)

# HDX reweighting
amide_resid = np.loadtxt('rewM9/amideresidues.txt')
fopen = np.load('rewM9/fopen.npy')


# load experimental data
exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

# split into blocks and perform reweighting
for time_factor in time_factors:

    window_ns = TAUR * time_factor
    # convert ns -> frames (frame_dt is in ps)
    frames_per_window = max(1, int(round(window_ns * 1000.0 / float(frame_dt_ps))))
    starts = list(range(0, n_frames - frames_per_window + 1, frames_per_window))
    block_data = []
    for iw, ws in enumerate(starts):
        we = ws + frames_per_window  # exclusive
        block = fopen[ws:we]   # shape (fw, n_bonds, 3)
        block_data.append(np.mean(block, axis=0))
    block_data = np.array(block_data)
    
    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(block_data)

    # reweight
    print(f'HDX reweighting with TAUR time factor of {time_factor} x ...')
    exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold)

    # save data
    stats_df.to_csv(save_dir+f'stats_df_{time_factor}xTAUR.csv')
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+f'wopt_array_{time_factor}xTAUR.npy',wopt_array)
    np.save(save_dir+f'wopt_{time_factor}xTAUR.npy',wopt)
    np.savetxt(save_dir+f'rew_lnPF_{time_factor}xTAUR.txt',rew_lnPF)
    np.savetxt(save_dir+f'prior_lnPF_{time_factor}xTAUR.txt',prior_lnPF)
    np.savetxt(save_dir+f'reweighting_chi2_phi_theta_{time_factor}xTAUR.txt',reweighting_data)




