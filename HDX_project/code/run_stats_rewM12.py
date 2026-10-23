import numpy as np
import sys
import os
from os import listdir
import pandas as pd
import mdtraj as mdj
import scipy.stats
import re
sys.path.insert(1, '/projects/lindorff_group/people/thd711/HDX_project/code/')

# parse input
input_proteins_file = sys.argv[1]
exp_PF_dir = sys.argv[2]

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

def find_xtc_filenames( path_to_dir, suffix=".xtc" ):
        filenames = listdir(path_to_dir)
        return [ filename for filename in filenames if filename.endswith( suffix ) ]

# read list of proteins to calculate PFs from a text file
with open(input_proteins_file,'r') as f:
    lines = f.readlines()

proteins = []
for line in lines:
    proteins.append(line.strip())

# forward model parameters dictionary
params = {'beta_c' : 0.35, 'beta_h' : 2.0}

# setup lists for final dataframe
PROTEIN = []
MODEL_RESIDUE = []
PF_EXP = []
PF_EXP_ERR = []
PF_SIM = []
PF_SIM_ELBOW = []
prior_PF_SIM = []

# match calculated and experimental values/residues
for protein in proteins:
    print(protein)
    # experimental data
    exp = pd.read_csv('EXP_DATA/{}/{}.csv'.format(exp_PF_dir, protein),index_col=0)
    model_residues = np.array(exp['Model_residue']).astype(int)
    expPF = np.array(exp['PF']).astype(float)
    expPF_error = np.array(exp['PF_error']).astype(float)
    
    # model pdb
    u = mdj.load('MD_DATA/{}/protein.pdb'.format(protein),top='MD_DATA/{}/protein.pdb'.format(protein))
    amides = u.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    amides_resnames = np.array([u.top.atom(i).residue.name for i in amides])
    amides_resids = np.array([u.top.atom(i).residue.resSeq for i in amides])

    # load amide res idxs
    amide_res_idx = np.loadtxt('MD_DATA/{}/rewM10/amideresidx.txt'.format(protein))
    assert amide_res_idx.shape[0]==amides_resids.shape[0]
    
    try:
        # load and average, reweighted data data
        lnPF = np.loadtxt('MD_DATA/{}/rewM12/log_lambda_rew_lnPF.txt'.format(protein))
        lnPF_elbow = np.loadtxt('MD_DATA/{}/rewM12/log_lambda_rew_lnPF_elbow.txt'.format(protein))
        prior_lnPF = np.loadtxt('MD_DATA/{}/rewM12/prior_lnPF.txt'.format(protein))
        expresids = np.loadtxt('MD_DATA/{}/rewM12/exp_residues.txt'.format(protein)).astype(int)
    except:
        print(f'No reweighting results for {protein}')
        continue

    # exp indices - filter experimental residues for which we had open state sampling and could do reweighting
    expidxs = []
    for ele in expresids:
        expidxs.append(list(model_residues).index(ele))
    expidxs=np.array(expidxs)
    model_residues = model_residues[expidxs]
    expPF = expPF[expidxs]
    expPF_error = expPF_error[expidxs]
    
    # calculate protection factors
    PF = np.exp(lnPF)
    PF_elbow = np.exp(lnPF_elbow)
    prior_PF = np.exp(prior_lnPF)

    # append all data to lists
    for i in range(len(model_residues)):
        PROTEIN.append(protein)
        MODEL_RESIDUE.append(model_residues[i])
        PF_EXP.append(float(expPF[i]))
        PF_EXP_ERR.append(float(expPF_error[i]))
        PF_SIM.append(PF[i])
        PF_SIM_ELBOW.append(PF_elbow[i])
        prior_PF_SIM.append(prior_PF[i])

# make final dataframe to save
df = pd.DataFrame()
df['Protein'] = PROTEIN
df['Model_residue'] = MODEL_RESIDUE
df['EXP PF'] = PF_EXP
df['EXP PF ERR'] = PF_EXP_ERR
df['SIM PF'] = PF_SIM
df['SIM PF ELBOW'] = PF_SIM_ELBOW
df['PRIOR SIM PF'] = prior_PF_SIM
df.to_csv('output_results/data_EXP_vs_SIM_rewM12_{}.csv'.format(exp_PF_dir))

# calculate RMSE in log10(PF) and Pearson correlation coefficient
x = np.log10(df['EXP PF'])
y = np.log10(df['SIM PF'])
rmse = RMSE(x,y)
pearson_r = scipy.stats.pearsonr(x,y)[0]

print('RMSE (log10 PF): {:.2f}'.format(rmse))
print('Pearson r: {:.2f}'.format(pearson_r))

