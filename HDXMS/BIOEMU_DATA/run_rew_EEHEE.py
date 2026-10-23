import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
bme_dir = '../code/'
sys.path.append(bme_dir)
from hdx_bme import HDXReweighterFast
from kint import *
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Process protein data")

# Add argument
parser.add_argument(
    "--protein_dir",                
    type=str,
    help="Protein directory"
)
parser.add_argument(
    "--save_dir",                 
    type=str,
    help="directory name"
)

# Parse arguments
args = parser.parse_args()

protein_dir = args.protein_dir
protein = protein_dir#.split('-')[0]
save_dir = args.save_dir
outpath = protein_dir + str('/') + save_dir + str("/")
print("Outpath: ", outpath)
print("Protein:", protein)

# get sequence  calculate kint
# conditions: 293.15 K, pH 6.17 (pH read), 0.863 D2O, 25mM MES
# pD = 6.515 in 86.3% D2O



with open(f'../EXP_DATA/{protein}.fasta', 'r') as f:
    lines =f.readlines()
seq = lines[-1].strip()
print(seq)

# kint in seconds - this setup of parameters matches the kint values calculated by Ferrari et al 
kint = Stofella2025(sequence = seq, temperature = 293.15, pH = 6.515, exchange_type='HD', hd_percentage=100.) 

# get experimental da1ta
df = pd.read_csv(f'../EXP_DATA/{protein}.csv')
F_exp = np.array(df['Rel. Uptake'])
# ensure all experimental values are between 0 and 1
F_exp = np.maximum(F_exp, float(0))
F_exp = np.minimum(F_exp, float(1))
print(np.min(F_exp))
print(np.max(F_exp))
sigma = np.abs(np.array(df['Rel. Uptake SD']))
times = np.array(df['Time (s)'])
peptides = []
for i in range(len(df)):
    pep_i = [idx for idx in range(int(df['Start'][i]) - 1, int(df['End'][i]))] # 0-indexed
    peptides.append(pep_i)

## load fopen

O_init = np.load(f'../BIOEMU_DATA/{protein_dir}/M14DH_scaled/O.npy')
res = np.loadtxt(f'../BIOEMU_DATA/{protein_dir}/M14DH_scaled/amideresidues.txt').astype(int)
residx = res-1

# fill proline/unmodelled positions with zeros
n_frames = O_init.shape[0]
n_residues = len(seq)

# Create output array full of zeros
O = np.zeros((n_frames, n_residues), dtype=O_init.dtype)

# Fill them
O[:, residx] = O_init
O.shape

# run optimisation theta scan and save results
thetas = np.geomspace(1e7, 1e0, 30)
# uniform prior
w0 = np.ones(O.shape[0])/O.shape[0]

wopt_array = [w0]
phis = []
rmses = []
chi2_values= []
pred_values = []
opt_conv = [True]


# reweighting object
# skip first two residues due to back-exchange as in Ferrari et al
reweighter = HDXReweighterFast(kint, O.T, peptides, times, F_exp, sigma, theta=thetas[0], n_skip=2) # O input expected as shape nres, nframes

# prior stats
chi2_values.append(reweighter.calc_chi2(w0))
rmses.append(reweighter.calc_rmse(w=w0))
phis.append(reweighter.calc_neff(w=w0, w0=w0))
pred_values.append(reweighter.predict_with_weights(w0))


# iterate through thetas
for theta in thetas:
    print('Working on theta ', theta)
    reweighter = HDXReweighterFast(kint, O.T, peptides, times, F_exp, sigma, theta=theta, n_skip=2) # O input expected as shape nres, nframes
    # Fit weights
    result = reweighter.fit()
    print(result)
    if not result.success:
        print(f"Optimization failed for theta={theta}: {result.message}, continuing...")
    
    # Predict 
    wopt = reweighter.w_opt
    # append data and save
    opt_conv.append(result.success)
    chi2_values.append(reweighter.calc_chi2(wopt))
    rmses.append(reweighter.calc_rmse(w=wopt))
    phis.append(reweighter.calc_neff(w=wopt, w0=w0))
    pred_values.append(reweighter.predict_with_weights(wopt))
    wopt_array.append(wopt)
    np.save(outpath+"opt_convergence.npy", np.array(opt_conv))
    np.save(outpath+"wopt_array.npy", np.array(wopt_array))
    np.savetxt(outpath+"chi2r.txt", np.array(chi2_values))
    np.savetxt(outpath+"rmse.txt", np.array(rmses))
    np.savetxt(outpath+"phis.txt", np.array(phis))
    np.save(outpath+"pred_uptakes.npy", np.array(pred_values))

print('Done')