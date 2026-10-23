import mdtraj as md
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
parser.add_argument("--w0", type=str, default=None, help="Optional filename with initial weights for reweighting. If not provided, uniform weights will be used.")
parser.add_argument("--charge_scaling", type=float, default=1.0, help="Scaling factor for charges. Default: 1.0")

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
        By default, all charges from the same amino acid and preceding residues are excluded here (because intrinsic HDX rates have been corrected for neighbour effects).
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
    charges = charges * E_CHARGE * charge_scaling # convert to Coulombs and apply scaling factor

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

def reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=None):

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
    if w0 is None:
        w0 = np.ones(n_frames) / n_frames  # uniform initial weights
    else:
        w0 = np.array(w0)
        w0 = w0 / np.sum(w0)  # normalise weights to sum to 1

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
    thetas = np.geomspace(1e6,1e0,50)
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
            # calculate electrostatics PF
            amide_resid, amide_res_idx, lnPF_DH = calc_PF_DH(trj, temperature=temperature, ionic=ionic, charge_scaling=charge_scaling_factor)
            # convert to fopen / DH_PF
            fopenDH = float(1.0) / np.exp(lnPF_DH)

        else:
            # run PF calculation
            # calculate electrostatics PF
            amide_resid2, amide_res_idx2, lnPF_DH = calc_PF_DH(trj, temperature=temperature, ionic=ionic, charge_scaling=charge_scaling_factor)
            # convert to fopen / DH_PF
            tmpfopenDH = float(1.0) / np.exp(lnPF_DH)
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
        print('Using provided initial weights for reweighting')
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:  
        w0 = None
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold)

    # save data
    stats_df.to_csv(save_dir+'stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+'wopt_array.npy',wopt_array)
    np.save(save_dir+'wopt.npy',wopt)
    np.savetxt(save_dir+'rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+'reweighting_chi2_phi_theta.txt',reweighting_data)

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
        print('Using provided initial weights for reweighting')
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0)
    else:  
        w0 = None
        exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold)

    # save data
    stats_df.to_csv(save_dir+'stats_df.csv')
    np.savetxt(save_dir+'exp_residues.txt',exp_residues)
    np.save(save_dir+'wopt_array.npy',wopt_array)
    np.save(save_dir+'wopt.npy',wopt)
    np.savetxt(save_dir+'rew_lnPF.txt',rew_lnPF)
    np.savetxt(save_dir+'prior_lnPF.txt',prior_lnPF)
    np.savetxt(save_dir+'reweighting_chi2_phi_theta.txt',reweighting_data)


