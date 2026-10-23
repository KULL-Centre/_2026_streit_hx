import mdtraj as md
import pandas as pd
import numpy as np
import sys
import os
import scipy.stats
bme_dir = '/projects/lindorff_group/people/thd711/HDX_project/code/BME/'
sys.path.append(bme_dir)
import BME as BME
import argparse
from scipy.spatial.distance import pdist

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--name", type=str, help="Name of the protein.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--max_frames", type=int, default=20001, help="Maximum number of frames to calculate data for per trajectory. Default: 20001")
parser.add_argument("--filter", type=str, default=False, help="Filter out frames with overlapping atoms (e.g. for BioEmu - not necessary for all-atom MD ensembles).")

args = parser.parse_args()

pdb = args.pdb
protname = args.name
save_dir = args.save_dir
max_frames_per_traj = args.max_frames
filter = args.filter

### PERFORM FORWARD MODEL CALCULATIONS ###

rho_threshold = 0.5
peptide_correction_factor = 0.957089 # correction factor based on model tripeptides

# load and re-save residue data
amide_resid = np.loadtxt(save_dir+'../M10/amideresidues.txt')
np.savetxt(save_dir+'amideresidues.txt', amide_resid)

# load descriptors for forward model
fopen = np.load(save_dir+'../M10/fopen.npy')
rho_front = np.load(save_dir+'../rho_front_noexcl/rho_front_frames.npy')
rho_open = (rho_front<rho_threshold)*1 # assign open state according to cutoff of rho_front 

O = fopen * rho_open # multiply SASA/hbond based open state with rho_front to get final open state definition for reweighting
O = O / peptide_correction_factor # apply peptide correction factor
np.save(save_dir+'O.npy', O) 
print('Frames x amides: ', O.shape)

        
