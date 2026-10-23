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
rew_method = args.rew_method
filter = args.filter

rho_threshold = 0.5
peptide_correction_factor = 0.951914 # correction factor based on model tripeptides

if rew_method=='log':
    save_prefix = 'log_'
elif rew_method=='log_lambda':
    save_prefix = 'log_lambda_'
elif rew_method=='linear':
    save_prefix = ''
else:
    print('Error: did not recognise reweighting method: ', rew_method)


### PERFORM FORWARD MODEL CALCULATIONS ###

# find all trajectories
trajs = sorted(find_xtc_filenames('./')) 


# prepare sim data
try:
    amide_resid = np.loadtxt(save_dir+'../rewM13DH_scaled/amideresidues.txt')
    fopenDH = np.load(save_dir+'../rewM13DH_scaled/fopenDH.npy').astype(float)
except:
    amide_resid = np.loadtxt(save_dir+'../rewM13DH_scaled_dGprior/amideresidues.txt')
    fopenDH = np.load(save_dir+'../rewM13DH_scaled_dGprior/fopenDH.npy').astype(float)
rho_front = np.load(save_dir+'../rho_front_noexcl/rho_front_frames.npy').astype(float)
np.savetxt(save_dir+'amideresidues.txt', amide_resid)
rho_open = (rho_front<rho_threshold)*1

O = fopenDH * rho_open # multiply SASA/hbond/electrostatics based open state with rho_front to get final open state definition for reweighting 
O = O / peptide_correction_factor # apply peptide correction factor
np.save(save_dir+'O.npy', O)
print('Frames x amides: ', O.shape)

sim_data = []
sim_data.append(amide_resid)
sim_data.append(O)

# load experimental data
exp = pd.read_csv('../../EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protname))

# reweight
print('Reweighting...')
if w0 is not None:
    print('Using provided initial weights for reweighting (--w0)')
else:
    print('Using uniform prior weights, no alternative prior weights were provided (--w0): w0=', w0)

thetas = list([1e5,1e4])+list(np.geomspace(1e3, 1e0, 50))

if rew_method=='linear':
    exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0, theta_grid=thetas)
elif rew_method=='log':
    exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_softmax(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0, theta_grid=thetas)
elif rew_method=='log_lambda':
    exp_residues, stats_df, wopt_array, wopt, reweighting_data, rew_lnPF, prior_lnPF = reweight_log_lambda(exp, sim_data, Neff_threshold=Neff_threshold, w0=w0, theta_grid=thetas)
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


