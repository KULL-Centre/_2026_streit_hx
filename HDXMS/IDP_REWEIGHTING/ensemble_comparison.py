import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
bme_dir = '/projects/lindorff_group/people/thd711/HDXMS/code/'
sys.path.append(bme_dir)
from hdx_bme import HDXReweighterFast
from kint import *
import argparse

# arguments: protein name, forward model name, ref_rates (Stofella2025 or Nguyen2018)
parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, help='Protein name')
parser.add_argument('--forward_model', type=str, help='Forward model name')
parser.add_argument('--ref_rates', type=str, help='Reference rates (Stofella2025 or Nguyen2018)')
parser.add_argument('--pD', type=float, default=7.4, help='pD value (default: 7.4)')
parser.add_argument('--temperature', type=float, default=293.15, help='Temperature in Kelvin (default: 293.15 K)')
parser.add_argument('--ensemble', required = True, type=str, help='Name of the type of ensemble to search for.')

args = parser.parse_args()

protein = args.protein
forward_model = args.forward_model
temperature = args.temperature
pD = args.pD
ensemble = args.ensemble
ref_rates = args.ref_rates
if ref_rates == 'Stofella2025':
    file_ending = ''
elif ref_rates == 'Nguyen2018':
    file_ending = '_PDLA'
else:
    raise ValueError("ref_rates must be either 'Stofella2025' or 'Nguyen2018'")


with open(f'../../../EXP_DATA/{protein}.fasta', 'r') as f:
    lines =f.readlines()
seq = lines[-1].strip()
print(seq)

print(f"pD: {pD}")

# kint in seconds

if ref_rates == 'Stofella2025':
    kint = Stofella2025(sequence = seq, temperature = temperature, pH = pD, exchange_type='HD', hd_percentage=100) 
elif ref_rates == 'Nguyen2018':
    kint = Nguyen2018(sequence = seq, temperature = temperature, pH = pD, exchange_type='HD', hd_percentage=100)
else:
    raise ValueError("ref_rates must be either 'Stofella2025' or 'Nguyen2018'")

print(kint)

# get experimental data
df = pd.read_csv(f'../../../EXP_DATA/{protein}.csv')
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

# load fopen
files = ['fopen.npy', 'fopenDH.npy', 'O.npy']
for file in files:
    try:
        O_init = np.load(f'../../../{ensemble}_IDPs/{protein}/{forward_model}/{file}')
        print(f"Loaded {file} from {forward_model}")
        break
    except FileNotFoundError:
        print(f"{file} not found in {forward_model}, trying next file.")

        
# fill proline positions with zeros
n_frames = O_init.shape[0]
n_residues = len(seq)

# Create output array full of zeros
O = np.zeros((n_frames, n_residues), dtype=O_init.dtype)

# Indices of non-proline residues
non_pro_idx = [i for i, aa in enumerate(seq) if aa != "P"]

# Fill them
O[:, non_pro_idx] = O_init
print('O.shape:', O.shape)

w0 = np.ones(O.shape[0])/O.shape[0]

rmses = []
chi2_values= []
pred_values = []


# reweighting object
# skip first two residues due to back-exchange as in Ferrari et al
reweighter = HDXReweighterFast(kint, O.T, peptides, times, F_exp, sigma, theta=1e3, n_skip=2) # O input expected as shape nres, nframes

# prior stats
chi2_values.append(reweighter.calc_chi2(w0))
rmses.append(reweighter.calc_rmse(w=w0))
pred_values.append(reweighter.predict_with_weights(w0))

# save data
np.savetxt(f"rmse{file_ending}.txt", rmses)
np.savetxt(f"chi2r{file_ending}.txt", chi2_values)
np.save(f"pred_uptakes{file_ending}.npy", np.array(pred_values))


print('Done')


rmses = np.loadtxt(f"rmse{file_ending}.txt")
chi2r = np.loadtxt(f"chi2r{file_ending}.txt")
pred_values = np.load(f"pred_uptakes{file_ending}.npy")


plt.figure(figsize=(4,4))
plt.scatter(F_exp, pred_values[0])
plt.xlabel('Exp')
plt.ylabel('Sim')
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.plot([-1, 4], [-1, 4])
plt.tight_layout()
plt.savefig(f'pred_vs_exp{file_ending}.png', dpi=200)

# plot protection factors
plt.figure(figsize=(4,3))
plt.plot(np.arange(1, len(seq)+1), 1/np.average(O, axis=0, weights=w0))
plt.axhline(1, color = 'grey')
plt.yscale('log')
plt.xlabel('Residue')
plt.ylabel('PF')
plt.tight_layout()
plt.savefig(f'protection_factors{file_ending}.png', dpi=200)


print(reweighter.calc_chi2(w0))
print(reweighter.calc_rmse(w=w0))

# plot uptake curves
reweighter.plot_uptake_curves(w0=w0, wopt=w0, savefile=f'pred_uptake_curves{file_ending}.png')