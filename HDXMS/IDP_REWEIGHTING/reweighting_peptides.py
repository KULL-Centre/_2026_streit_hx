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

args = parser.parse_args()

protein = args.protein
forward_model = args.forward_model
ref_rates = args.ref_rates
if ref_rates == 'Stofella2025':
    file_ending = ''
elif ref_rates == 'Nguyen2018':
    file_ending = '_PDLA'
else:
    raise ValueError("ref_rates must be either 'Stofella2025' or 'Nguyen2018'")

# conditions for Angiotensin, Bradykinin
# 293 K, pH read 7.0, 95% D2O, ion strength 0.02 M

with open(f'../../../EXP_DATA/{protein}.fasta', 'r') as f:
    lines =f.readlines()
seq = lines[-1].strip()
print(seq)

pHr = 7.0
d_frac = 0.95
pD = pHr + 0.4 * d_frac
print(f"pD: {pD}")

# kint in seconds

if ref_rates == 'Stofella2025':
    kint = Stofella2025(sequence = seq, temperature = 293.15, pH = pD, exchange_type='HD', hd_percentage=100) 
elif ref_rates == 'Nguyen2018':
    kint = Nguyen2018(sequence = seq, temperature = 293.15, pH = pD, exchange_type='HD', hd_percentage=100)
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
        O_init = np.load(f'../../../MD_DATA/{protein}/{forward_model}/{file}')
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

# run optimisation theta scan and save results
thetas = np.geomspace(1e7, 1e0, 30)
# uniform prior
w0 = np.ones(O.shape[0])/O.shape[0]

wopt_array = [w0]
phis = []
rmses = []
chi2_values= []
pred_values = []


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
    reweighter = HDXReweighterFast(kint, O.T, peptides, times, F_exp, sigma, theta=theta, n_skip=2) # O input expected as shape nres, nframes
    # Fit weights
    result = reweighter.fit()
    print(f"theta={theta}")
    print(" success:", result.success)
    #print(" status :", getattr(result, "status", None))
    #print(" message:", getattr(result, "message", None))
    #print(" fun    :", getattr(result, "fun", None))
    #print(" nit    :", getattr(result, "nit", None))
    #print(" nfev   :", getattr(result, "nfev", None))
    #print(result)
    if not result.success:
        print('minimisation not converged, continuing to next theta')
        continue
    # Predict 
    wopt = reweighter.w_opt
    # append data and save
    chi2_values.append(reweighter.calc_chi2(wopt))
    rmses.append(reweighter.calc_rmse(w=wopt))
    curr_phi = reweighter.calc_neff(w=wopt, w0=w0)
    phis.append(curr_phi)
    pred_values.append(reweighter.predict_with_weights(wopt))
    wopt_array.append(wopt)
    np.save(f"wopt_array{file_ending}.npy", np.array(wopt_array))
    np.savetxt(f"chi2r{file_ending}.txt", np.array(chi2_values))
    np.savetxt(f"rmse{file_ending}.txt", np.array(rmses))
    np.savetxt(f"phis{file_ending}.txt", np.array(phis))
    np.save(f"pred_uptakes{file_ending}.npy", np.array(pred_values))

    if curr_phi < 0.001:
        print('phi below 0.001, stopping scan')
        break

print('Done')


phis = np.loadtxt(f"phis{file_ending}.txt")
rmses = np.loadtxt(f"rmse{file_ending}.txt")
chi2r = np.loadtxt(f"chi2r{file_ending}.txt")
wopt_array = np.load(f"wopt_array{file_ending}.npy")
pred_values = np.load(f"pred_uptakes{file_ending}.npy")

# find index of last neff above 0.2
opt_idx = np.where(phis>0.2)[0][-1]
print(opt_idx)
    
# Left axis
plt.figure(figsize=(6,4))
protein = 'Bradykinin'
ax = plt.gca()
ax.plot(phis, chi2r, 'o-', color='tab:red')
ax.set_ylabel("$\chi$$_{r}$$^{2}$", color='tab:red')
ax.tick_params(axis='y', labelcolor='tab:red')
ax.axvline(phis[opt_idx], color = 'grey', linestyle='--')
ax.set_title(protein)
ax.set_xlabel("$\phi$")
ax.set_ylim(0,)

# Right axis
ax2 = ax.twinx()
ax2.plot(phis, rmses, 'o-', color='tab:purple')
ax2.set_ylabel("RMSE", color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')
ax2.set_ylim(0,)


plt.tight_layout()
plt.savefig(f'Lcurve{file_ending}.png', dpi=200)


plt.figure(figsize=(4,4))
plt.scatter(F_exp, pred_values[0])
plt.scatter(F_exp, pred_values[opt_idx])
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
plt.plot(np.arange(1, len(seq)+1), 1/np.average(O, axis=0, weights=wopt_array[opt_idx]))
plt.axhline(1, color = 'grey')
plt.yscale('log')
plt.xlabel('Residue')
plt.ylabel('PF')
plt.tight_layout()
plt.savefig(f'protection_factors{file_ending}.png', dpi=200)

print(reweighter.calc_chi2(w0))
print(reweighter.calc_chi2(wopt_array[opt_idx]))
print(reweighter.calc_rmse(w=w0))
print(reweighter.calc_rmse(w=wopt_array[opt_idx]))
print(reweighter.calc_neff(w=wopt_array[opt_idx], w0=w0))

# plot uptake curves
reweighter.plot_uptake_curves(w0=w0, wopt=wopt_array[opt_idx], savefile=f'pred_uptake_curves{file_ending}.png')