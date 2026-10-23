from simtk import unit, openmm
from simtk.openmm.app import PDBFile, ForceField, Modeller
import json
import mdtraj as md
import pandas as pd
import numpy as np
import argparse
from rew_functions import *
from utils import *

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--topology", type=str, help="Path to the topology file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in Kelvin. Default: 300.0")
parser.add_argument("--ionic", type=float, default=0.15, help="Ionic concentration. Default: 0.15")
parser.add_argument("--min_neff", type=float, default=0.1, help="Minimum threshold for the fraction of effective frames. Default: 0.1")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")
parser.add_argument("--w0", type=str, default=None, help="Optional filename with initial weights for reweighting. If not provided, uniform weights will be used.")
parser.add_argument("--charge_scaling", type=float, default=1.0, help="Scaling factor for charges. Default: 1.0")
parser.add_argument('--rew_method', type=str, default='linear', 
            help='Reweighting method: linear (fraction open space using BME/Lagrange multipliers), log (ln PF space directly over weights), log_lambda (ln PF space over lambda parameters)')
parser.add_argument("--filter", action="store_true", help="Filter out frames with overlapping atoms (e.g. for BioEmu - not necessary for all-atom MD ensembles).")


args = parser.parse_args()

pdb = args.pdb
topfile = args.topology
protname = args.name
save_dir = args.save_dir
temperature = args.temperature
ionic = args.ionic
Neff_threshold = args.min_neff
max_frames_per_traj = args.max_frames
w0 = np.loadtxt(args.w0) if args.w0 else None
charge_scaling_factor = args.charge_scaling
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

# function to calculate protection factor contribution from electrostatics
def calc_PF_DH(trj, topfile = topfile, temperature = temperature, ionic = ionic, charge_scaling = 1.0):

    # Constants
    EPSILON_0 = 8.854187817e-12 # Vacuum permittivity (F/m or C^2/(N m^2))
    E_CHARGE = 1.602176634e-19  # Elementary charge (C)
    NA = 6.02214076e23          # Avogadro's number
    kB = 1.380649e-23           # J/K (Boltzmann constant)
    F = 9.648533e4              # Faraday constant, C/mol
    z = -1.0                    # charge of hydroxide ion
    R = 8.314463                # Gas constant, J K-1 mol-1

    def debye_length(ionic_strength, temperature=298.15):

        # ionic strength input in units of M (molar)

        # conversions
        Im = ionic_strength * 1e3  # mol/m^3

        # dielectric
        # temperature dependence of dielectric constant using the empirical formula proposed by Akerlof & Oshry (1950)
        dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
        
        numerator = dielectric * EPSILON_0 * kB * temperature
        denominator = 2 * NA * (E_CHARGE ** 2) * Im
        
        lambda_D = np.sqrt(numerator / denominator)  # in meters
        
        return lambda_D

    def calculate_debye_huckel_potential(positions, amideH_indices, residues, chains, charges, debye_len, temperature=298.15):
        """
        Calculate Debye-Hückel potential at each amide hydrogen atom.
        By default, all charges from the same amino acid and preceding residue are excluded here (because intrinsic HDX rates have been corrected for neighbour effects).
        
        Parameters:
        - positions: Nx3 array in nanometers
        - charges: length-N array in elementary charges
        - debye_len: Debye length in meters
        - dielectric: relative permittivity
        
        Returns:
        - potentials: length-N array of potentials in Volts
        """
        N = len(positions)
        potentials = []

        # calculate dielectric
        # temperature dependence of dielectric constant using the empirical formula proposed by Akerlof & Oshry (1950)
        dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
        
        for idx in amideH_indices:
            r_i = np.array([positions[idx]])
            residue_i = residues[idx]
            chain_i = chains[idx]

            # boolean matrix whether or not to exclude the pair-wise contributions from the same or neighbouring residues
            # 1 = used
            # 0 = excluded
            bool_res = ~((residues == residue_i) | (residues == residue_i - 1)) # exclude preceding and current residue
            bool_res = bool_res.astype(int)
            bool_chain = ( np.abs(chain_i - chains) > 0) * 1 # keep if residue in a different chain
            bool = bool_res + bool_chain # i.e., both the residue and chain criteria. 0 = exlude; 1,2 = keep
            keepidxs = np.where(bool > 0)[0]
        
            # calculate distances in nanometres
            r = np.linalg.norm( r_i - positions , axis=1)
            
            # convert to metres
            r = r[keepidxs] * 1e-9

            # extract final charges
            q = charges[keepidxs]

            # calculate potential
            pot_i = np.sum( q * np.exp(-r / debye_len) / r)

            # include physical constants
            pot_i = pot_i * ( 1 / (4 * np.pi * EPSILON_0 * dielectric ) )

            potentials.append(pot_i)
        
        # return potentials
        return np.array(potentials)


    # function to obtain atomic partial charges from a force field
    # this function expects the input structures to be protonated
    def get_charges(pdb_file, forcefield='amber14'):
        
        # Load structure
        pdb = PDBFile(pdb_file)
        
        # Load force field
        ff = ForceField('{}-all.xml'.format(forcefield), '{}/tip3p.xml'.format(forcefield))
        
        # Create Modeller for adding hydrogens if needed
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Create system to assign forces and charges
        system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff)
        
        # Extract partial charges from system
        charges = []
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nbforce = force
                break
        
        for i in range(system.getNumParticles()):
            charge, sigma, epsilon = nbforce.getParticleParameters(i)
            charges.append(charge._value)  # charges are in elementary charge units
        
        charges = np.array(charges)

        return charges
    
    def get_charge_dict(json_file):
        """Return (residue_index, atom_name) list to uniquely identify atoms."""

        # Load the JSON file as a dictionary
        with open(json_file, 'r') as f:
            charge_dict = json.load(f)

        return charge_dict
    
    def get_charges2(pdb_filename,json_file='/projects/lindorff_group/people/thd711/HDX_project/code/partial_charges.json'):

        # get charge dictionary
        charge_dict = get_charge_dict(json_file)

        # Load original structure and get residue/atom information
        # Load the PDB file
        pdb = PDBFile(pdb_filename)

        # Get the topology
        topology = pdb.topology

        # List to store atom information
        residue_numbers = []
        residue_names = []
        atom_names = []

        # Iterate through all atoms in the topology
        for atom in topology.atoms():
            residue = atom.residue
            residue_numbers.append(int(residue.id)) # Actual residue number from PDB file
            residue_names.append(residue.name) # Residue name (e.g., ALA, GLY)
            atom_names.append(atom.name) # Atom name (e.g., CA, N)
            
        # adjust names of terminal residues with "N" and "C"
        final_residue_names = []
        start_residue = np.min(residue_numbers)
        end_residue = np.max(residue_numbers)

        for resname,resid in zip(residue_names,residue_numbers):
            if resid==start_residue:
                final_residue_names.append(str('N')+resname)
            elif resid==end_residue:
                final_residue_names.append(str('C')+resname)
            else:
                final_residue_names.append(resname)
        
        # list of partial charges
        charges = []
        for resname,atom,resnum in zip(final_residue_names,atom_names,residue_numbers):

            if resname[0]=='N' and atom=='H': # first amine hydrogen atom in chain needs to be renamed
                atom='H1'

            if resname=='HIS':
                try:
                    ch = float(charge_dict['HID'][atom])
                except:
                    ch = float(charge_dict['HIE'][atom])
            else:
                ch = float(charge_dict[resname][atom])

            charges.append(ch)

        return np.array(charges)


    def phi_to_lnPF(potential, temperature = 298.15):
        
        # Calculate the PF from the electrostatic potential given in Volts

        return ( (z * F * potential) / ( R * temperature ) )


    ########################
    ### RUN THE ANALYSIS ###
    ########################
        
    # load trajectory
    trj = trj
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

    # check how many protein chains and if the chains are identical
    nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains

    # obtain the charges and convert to Coulomb
    try:
        charges = get_charges(topfile)
    except:
        charges = get_charges2(topfile) # alternative way to get charges if openmm fails
    charges = charges * E_CHARGE * charge_scaling # convert to Coulomb and apply scaling factor

    # obtain residues and chain ids for all atoms
    all = trj.top.select("protein or resname NH2 or resname ACE or resname NME or type Ca or type Mg")
    residues = np.array([trj.top.atom(i).residue.resSeq for i in all]).astype(int)
    chainids = []

    if nprot_chains==1:
        
        for res in residues:
            chainids.append(0)

    elif nprot_chains==2: # if two chains

        for i in range(len(residues)):
            if i < int(len(residues)/2):
                chainids.append(0)
            else:
                chainids.append(1)

    else:
        print('Error: more than two chains are not supported')

    chainids = np.array(chainids).astype(int)

    # get amide hydrogen indices
    amideH_indices = trj.top.select('(name H or name HN) and protein')

    # get debye_length / salt screening parameter
    debye_len = debye_length(ionic_strength = ionic, temperature = temperature)

    # list for frame-wise potentials
    phis = []

    for frame in range(len(trj)):

        # get positions
        positions = trj.xyz[frame][all] # all selected indices belong to protein etc

        phi = calculate_debye_huckel_potential(positions, amideH_indices, residues, chainids, charges, debye_len, temperature = temperature)

        phis.append(phi)

    # convert to array and PFs
    phis = np.array(phis)

    # convert to lnPF per frame
    lnPF_frames = phi_to_lnPF(phis, temperature = temperature)
    
    return amide_resid, amide_res_idx, lnPF_frames

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
            amide_resid, amide_res_idx, bool_open = calc_M9(trj)
            # calculate electrostatics PF
            amide_resid2, amide_res_idx2, lnPF_DH = calc_PF_DH(trj, temperature=temperature, ionic=ionic, charge_scaling=charge_scaling_factor)
            # make sure same residues were calculated
            assert len(amide_resid)==len(amide_resid2)
            assert np.sum(np.array(amide_resid)-np.array(amide_resid2))==0
            # convert to fopen / DH_PF
            fopenDH = bool_open.astype(float) / np.exp(lnPF_DH)

        else:
            # run PF calculation
            # calculate open closed
            amide_resid, amide_res_idx, bool_open = calc_M9(trj)
            # calculate electrostatics PF
            amide_resid2, amide_res_idx2, lnPF_DH = calc_PF_DH(trj, temperature=temperature, ionic=ionic, charge_scaling=charge_scaling_factor)
            # make sure same residues were calculated
            assert len(amide_resid)==len(amide_resid2)
            assert np.sum(np.array(amide_resid)-np.array(amide_resid2))==0
            # convert to fopen / DH_PF
            tmpfopenDH = bool_open.astype(float) / np.exp(lnPF_DH)
            # append
            fopenDH = np.concatenate((fopenDH, tmpfopenDH),axis=0)

    # save frame-wise data
    np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
    np.savetxt(save_dir+'amideresidues.txt',amide_resid)
    np.save(save_dir+'fopenDH.npy',fopenDH)

    # prepare sim data
    amide_resid = np.loadtxt(save_dir+'amideresidues.txt')
    fopenDH = np.load(save_dir+'fopenDH.npy')

    # prepare sim data
    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(fopenDH)

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
    fopenDH = np.load(save_dir+'fopenDH.npy')

    sim_data = []
    sim_data.append(amide_resid)
    sim_data.append(fopenDH)

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


