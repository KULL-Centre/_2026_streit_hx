import mdtraj as md
import MDAnalysis as mda
import pandas as pd
import numpy as np
from simtk import unit, openmm
from simtk.openmm.app import PDBFile, ForceField, Modeller
import json
import sys
import os
import scipy.stats
bme_dir = '/projects/lindorff_group/people/thd711/HDX_project/code/BME/'
sys.path.append(bme_dir)
import BME as BME
import argparse

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--topology", type=str, help="Path to the topology file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in Kelvin. Default: 300.0")
parser.add_argument("--ionic", type=float, default=0.15, help="Ionic concentration. Default: 0.15")
parser.add_argument("--min_neff", type=float, default=0.1, help="Minimum threshold for the fraction of effective frames. Default: 0.1")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")

args = parser.parse_args()

pdb = args.pdb
topfile = args.topology
protname = args.name
save_dir = args.save_dir
temperature = args.temperature
ionic = args.ionic
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

# switching function for p(open | mindist)
def switch_mindist(x, alpha = 1.0, k = 5.17, x0 = 3.24, min_threshold = 0.1):

        arr = alpha / (1 + np.exp(-k * (x - x0)))
        return np.where(arr >= min_threshold, arr, float(0.0))

# switching function for p(open | SASA)
# stretched exponential
def switch_sasa(x, alpha = 1.0, x0 = 1981.87, beta = 736.87, min_threshold = 0.1):

        arr = alpha * (1 - np.exp( - ( x / x0 ) * beta ) ) 
        return np.where(arr >= min_threshold, arr, float(0.0))

# function to calculate open/closed status
def calc_mindist_sasa(trj):

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

    # calculate fraction open - p(open | descriptor)
    # using empirically determined switching functions
    #fopen_mindist = switch_mindist(min_distances, alpha = 1.0, k = k, x0 = x0, min_threshold=0.1)
    #fopen_sasa = switch_sasa(amideH_SASA, alpha = 1.0, x0 = sasx0, beta = beta, min_threshold=0.1)

    return amide_resid, amide_res_idx, min_distances, amideH_SASA

# function to calculate protection factor contribution from electrostatics
def calc_PF_DH(trj, topfile = topfile, temperature = temperature, ionic = ionic):

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

    def calculate_debye_huckel_potential(positions, amideH_indices, residues, chains, charges, debye_len, exclude = 1, temperature=298.15):
        """
        Calculate Debye-Hückel potential at each amide hydrogen atom.
        By default, all charges from the same amino acid and neighbouring residues are excluded here (because intrinsic HDX rates have been corrected for neighbour effects).
        This is defined by the "exclude" parameter.
        
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
            bool_res = ( np.abs(residue_i - residues) > exclude) * 1 # keep if separated more by more than number of exclusion residues
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


    # run the overall charge analysis
        
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
    charges = charges * E_CHARGE

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

        phi = calculate_debye_huckel_potential(positions, amideH_indices, residues, chainids, charges, debye_len, exclude = 1, temperature = temperature)

        phis.append(phi)

    # convert to array and PFs
    phis = np.array(phis)

    # convert to lnPF per frame
    lnPF_frames = phi_to_lnPF(phis, temperature = temperature)
    
    return amide_resid, amide_res_idx, lnPF_frames

def reweight(exp, sim_data, Neff_threshold=Neff_threshold):

    # simulated data
    sim_res = sim_data[0]
    sim_calc = sim_data[1] # fopen / DH_PF

    # filter out residues that have an average fopenDH = 0 (i.e. no open state sampled)
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
    thetas = np.geomspace(1e10,1e0,50)
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
    if len(final_phis)==0:
        fine_thetas = np.geomspace(1e0, 1e-2, 50)
        for theta in fine_thetas:
            chi2_before, chi2_after, phi = rew_obj.fit(theta=theta)
            if phi < Neff_threshold - neff_diff_tol:
                break
            if chi2_after > 0.0:
                wopt_array.append(rew_obj.get_weights())
                final_thetas.append(theta)
                final_phis.append(phi)
                final_chi2.append(chi2_after)
    elif np.min(final_phis) > Neff_threshold + neff_diff_tol:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-2, 50)
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
    if len(final_phis)==0:
        fine_thetas = np.geomspace(1e-2, 1e-6, 50)
        for theta in fine_thetas:
            chi2_before, chi2_after, phi = rew_obj.fit(theta=theta)
            if phi < Neff_threshold - neff_diff_tol:
                break
            if chi2_after > 0.0:
                wopt_array.append(rew_obj.get_weights())
                final_thetas.append(theta)
                final_phis.append(phi)
                final_chi2.append(chi2_after)
    elif np.min(final_phis) > Neff_threshold + neff_diff_tol:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-6, 50)
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

def calc_prot_hbonds(trj, threshold_hbond = 0.24, treshold_angle = 150.0):
    
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices and residue numbers
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

    CAall = trj.top.select('name CA or name CH3 or (name N and resname NH2)') # CH3 in case of ACE and NH2 nitrogen atom in C-terminal NH2 resid
    all_resid = [trj.top.atom(i).residue.resSeq for i in CAall]

    # check how many protein chains and if the chains are identical
    nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains
    
    hbonds_res = np.zeros((len(trj), len(amide_res_idx)))
    
    if nprot_chains==1:

        for i, resid in enumerate(amide_resid):
       
            # additional hydrogen bond acceptors other than O
            other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
            
            amideN_idx = trj.top.select(f"protein and name N and resSeq {resid}")        
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

                # additional hydrogen bond acceptors other than O
                other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
                
                amideN_idx = np.array([trj.top.select(f"name N and resSeq {resid}")[0]]) # first match, first chain
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
                # additional hydrogen bond acceptors other than O
                other_hbonds = f" or protein and symbol F or (protein and symbol N and (resSeq != {resid} or name != N))"
                
                amideN_idx = np.array([trj.top.select(f"name N and resSeq {resid}")[1]]) # second match, second chain
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

    # convert hbond status to fopen
    fopen_hbonds = (hbonds_res==0)*1

    return amide_resid, amide_res_idx, fopen_hbonds

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

### END OF FUNCTIONTS ###
### END OF FUNCTIONTS ###
### END OF FUNCTIONTS ###

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
amide_resid = np.loadtxt('rewM10/amideresidues.txt')
mindist = np.load('rewM10/mindist.npy')
sasa = np.load('rewM10/sasa.npy')
# convert to fopen
fopen_mindist = switch_mindist(mindist)
fopen_sasa = switch_sasa(sasa)
fopen_hbonds = np.load('rewM10/fopen_hbonds.npy')
lnPF_DH = np.load('rewM10/lnPF_DH.npy')

descriptors = np.array([ fopen_mindist, fopen_sasa, fopen_hbonds])
fopen = np.min(descriptors, axis=0)
fopenDH = fopen.astype(float) / np.exp(lnPF_DH)
assert fopen.shape[1] == len(amide_resid)


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
        block = fopenDH[ws:we]   # shape (fw, n_bonds, 3)
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




