import mdtraj as mdj
import numpy as np
import sys

pdb = str(sys.argv[1])
traj = str(sys.argv[2])
save_dir = str(sys.argv[3])

def calc_M8_M9(trj, beta_s = 1.3e-3,  gamma_s = 2.64, beta_p = 3.65e1, gamma_p = 1.27, sasa_thresholds = [1.0, 2.5, 4.0, 5.5, 7.0], 
               distance_threshold = 3.0):

    """
    Compute protection factors according to Models M8 and M9 by Mohammadiarani et al 2018.
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

    # split trajectory into 10 blocks if nframes > 20000
    amideH_SASA = np.array([])

    if len(trj)>30000:
        nframes = len(trj)
        block_size = int(nframes/10)
        nblocks = int(nframes/block_size)
        for i in range(nblocks):
            if i==nblocks-1:
                sasa = mdj.shrake_rupley(subtraj[i*block_size:], probe_radius=0.14, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
            else:
                sasa = mdj.shrake_rupley(subtraj[i*block_size:(i+1)*block_size], probe_radius=0.14, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
            if i==0:
                amideH_SASA = sasa[:,amide_idxs]
            else:
                amideH_SASA = np.concatenate((amideH_SASA, sasa[:,amide_idxs]),axis=0)

            
    else:
        sasa = mdj.shrake_rupley(subtraj, probe_radius=0.14, n_sphere_points=960, mode='atom')*100 # convert nm^2 to Angstrom^2
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
            distances = mdj.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
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
                distances = mdj.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
                min_distances.append(np.min(distances, axis=1))

            else: # second chain

                remove_closest = f" and not (chainid 1 and resSeq {max(np.min(amide_resid),resid-1)} to {min(np.max(amide_resid),resid+1)})"
                amideH_idx = np.array([trj.top.select(f"(name H or name HN) and resSeq {resid}")[1]]) # second match, second chain
                polar_idx = trj.top.select("protein and (symbol O or symbol N or symbol S or symbol F)" + remove_closest)
                atom_pairs = []
                for idx in polar_idx:
                    atom_pairs.append([amideH_idx[0],idx])
                atom_pairs = np.array(atom_pairs)
                distances = mdj.compute_distances(trj, atom_pairs)*10 # convert nm to Angstrom
                min_distances.append(np.min(distances, axis=1))

    else:
        print('Error: more than two chains are not supported')


    min_distances = np.array(min_distances)
    min_distances = min_distances.T

    # calculate protection factors M8 (in the original paper they average SASA and mindist first before calculating lnPF)
    mean_SASA = np.maximum(np.mean(amideH_SASA,axis=0),10/len(trj)) # assuming at least one open state with amide H SASA of 10 Angstrom^2
    mean_mindist = np.mean(min_distances,axis=0)
    M8lnPF = beta_s * (1 / (mean_SASA ** gamma_s)) + beta_p * (1 / (mean_mindist ** gamma_p))

    # calculate protection factors M9
    # open state corresponds to amides exceding a sasa_threshold and having no polar protein atom within the distance threshold
    M9lnPF_list = []
    n_open_list = []
    n_closed_list = []
    for sasa_threshold in sasa_thresholds:
        bool_SASA = (amideH_SASA > sasa_threshold)*1
        bool_dist = (min_distances > distance_threshold)*1
        bool_sum = bool_SASA + bool_dist
        bool_open = (bool_sum==2)*1 # both conditions fulfilled
        n_open = np.sum(bool_open,axis=0)
        n_closed = len(trj) - n_open # length of trajectory = total number of snapshots
        PF = 1 + n_closed / np.maximum(n_open,float(1)) # np.maximum prevents from having 0 open/closed states, assuming at least 1 open state in trajectory
        M9lnPF = np.log(PF)
        M9lnPF_list.append(M9lnPF)
        n_open_list.append(n_open)
        n_closed_list.append(n_closed)

    # partial components
    inverse_sasa = 1 / amideH_SASA
    inverse_mindist = 1 / min_distances
    inverse_sasa = np.mean(inverse_sasa,axis=0)
    inverse_mindist = np.mean(inverse_mindist,axis=0)

    inverse_sasa_pow2 = 1 / amideH_SASA**2
    inverse_mindist_pow2 = 1 / min_distances**2
    inverse_sasa_pow2 = np.mean(inverse_sasa_pow2,axis=0)
    inverse_mindist_pow2 = np.mean(inverse_mindist_pow2,axis=0)

    inverse_sasa_pow05 = 1 / amideH_SASA**0.5
    inverse_mindist_pow05 = 1 / min_distances**0.5
    inverse_sasa_pow05 = np.mean(inverse_sasa_pow05,axis=0)
    inverse_mindist_pow05 = np.mean(inverse_mindist_pow05,axis=0)

    # save data
    np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
    np.savetxt(save_dir+'M8lnPF_{}.txt'.format(traj[:-4]),M8lnPF)
    np.savetxt(save_dir+'sasa_{}.txt'.format(traj[:-4]),mean_SASA)
    np.savetxt(save_dir+'mindist_{}.txt'.format(traj[:-4]),mean_mindist)
    np.savetxt(save_dir+'inverse_sasa_{}.txt'.format(traj[:-4]),inverse_sasa)
    np.savetxt(save_dir+'inverse_mindist_{}.txt'.format(traj[:-4]),inverse_mindist)
    np.savetxt(save_dir+'inverse_sasa_pow2_{}.txt'.format(traj[:-4]),inverse_sasa_pow2)
    np.savetxt(save_dir+'inverse_mindist_pow2_{}.txt'.format(traj[:-4]),inverse_mindist_pow2)
    np.savetxt(save_dir+'inverse_sasa_pow05_{}.txt'.format(traj[:-4]),inverse_sasa_pow05)
    np.savetxt(save_dir+'inverse_mindist_pow05_{}.txt'.format(traj[:-4]),inverse_mindist_pow05)

    for sasa_threshold,IDX in zip(sasa_thresholds,range(len(sasa_thresholds))):
        np.savetxt(save_dir+'M9lnPF_sasa_{}Ang_{}.txt'.format(sasa_threshold,traj[:-4]),M9lnPF_list[IDX])
        np.savetxt(save_dir+'n_open_sasa_{}Ang_{}.txt'.format(sasa_threshold,traj[:-4]),n_open_list[IDX])
        np.savetxt(save_dir+'n_closed_sasa_{}Ang_{}.txt'.format(sasa_threshold,traj[:-4]),n_closed_list[IDX])

    # no return in this case
    #return amide_res_idx, amide_resid, M8lnPF, M9lnPF_list[0], M9lnPF_list[1], M9lnPF_list[2], inverse_sasa, inverse_mindist, inverse_sasa_pow2, inverse_mindist_pow2, inverse_sasa_pow05, inverse_mindist_pow05, n_open, n_closed


# running
trj = mdj.load(traj,top=pdb)
calc_M8_M9(trj)
#amide_res_idx, amide_resid, M8lnPF, M9lnPF, inverse_sasa, inverse_mindist, inverse_sasa_pow2, inverse_mindist_pow2, inverse_sasa_pow05, inverse_mindist_pow05, n_open, n_closed = calc_M8_M9(trj)


