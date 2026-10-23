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
beta_c_vals = np.linspace(0,2,100)
beta_h_vals = np.linspace(0,15,100)

# matrices with statistics
stats_matrix_rmse = np.zeros((100,100))
stats_matrix_pearsonr = np.zeros((100,100))

# setup lists for final dataframe with optimal parameters
PROTEIN = []
MODEL_RESIDUE = []
PF_EXP = []
PF_EXP_ERR = []
C_SIM = []
C_SIM_ERR = []
HB_SIM = []
HB_SIM_ERR = []

# match calculated and experimental values/residues
for protein in proteins:
    print(protein)
    # experimental data
    exp = pd.read_csv('EXP_DATA/{}/{}.csv'.format(exp_PF_dir, protein),index_col=0)
    model_residues = np.array(exp['Model_residue']).astype(int)
    
    # model pdb
    u = mdj.load('MD_DATA/{}/protein.pdb'.format(protein),top='MD_DATA/{}/protein.pdb'.format(protein))
    amides = u.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    amides_resnames = np.array([u.top.atom(i).residue.name for i in amides])
    amides_resids = np.array([u.top.atom(i).residue.resSeq for i in amides])

    # load amide res idxs
    amide_res_idx = np.loadtxt('MD_DATA/{}/BV2006/amideresidx.txt'.format(protein))
    assert amide_res_idx.shape[0]==amides_resids.shape[0]
    
    exp_idxs = []
    for res in model_residues:
        hits = np.where(amides_resids==res)[0]
        if len(hits)==1:
            exp_idxs.append(hits[0])
        elif len(hits)>1:
            exp_idxs.append([hit for hit in hits])
        elif len(hits)==0:
            print('Warning: there is no match for residue {} in the pdb file of {}'.format(res,protein))
        
    exp_idxs = np.array(exp_idxs).astype(int)
    
    # treat monomer and dimer differently
    try:
        n_prot = exp_idxs.shape[1]
    except:
        n_prot=1

    # check how many simulations/files there are
    trajs = find_xtc_filenames('MD_DATA/{}/'.format(protein))
    n_trajs = len(trajs)

    # load and average data
    tmphb = []
    tmpc = []
    for traj in trajs:
        tmphb.append(np.loadtxt('MD_DATA/{}/BV2006/hbonds_res_{}.txt'.format(protein,traj[:-4])))
        tmpc.append(np.loadtxt('MD_DATA/{}/BV2006/contacts_res_{}.txt'.format(protein,traj[:-4])))
    tmphb = np.array(tmphb)
    tmpc = np.array(tmpc)

    hb = np.mean(tmphb,axis=0)
    c = np.mean(tmpc,axis=0)
    hb_err = np.std(tmphb,axis=0)/np.sqrt(n_trajs)
    c_err = np.std(tmpc,axis=0)/np.sqrt(n_trajs)

    # monomer vs dimer
    if n_prot==1: # monomer
        hb = hb[exp_idxs]
        c = c[exp_idxs]
        hb_err = hb_err[exp_idxs]
        c_err = c_err[exp_idxs]

    elif n_prot==2: # dimer - average the chains
        exp_idxs_1 = exp_idxs[:,0]
        exp_idxs_2 = exp_idxs[:,1]
        assert exp_idxs_1.shape[0]==exp_idxs_2.shape[0]

        hb1 = hb[exp_idxs_1]
        c1 = c[exp_idxs_1]
        hb_err1 = hb_err[exp_idxs_1]
        c_err1 = c_err[exp_idxs_1]

        hb2 = hb[exp_idxs_2]
        c2 = c[exp_idxs_2]
        hb_err2 = hb_err[exp_idxs_2]
        c_err2 = c_err[exp_idxs_2]

        # average the chains
        hb = hb1*0.5+hb2*0.5
        c = c1*0.5+c2*0.5
        hb_err = np.sqrt(hb_err1**2+hb_err2**2)/2
        c_err = np.sqrt(c_err1**2+c_err2**2)/2

    else:
        break
        print('Error: more than two protein copies are not yet supported')

    # append all data to lists
    for i in range(len(model_residues)):
        PROTEIN.append(protein)
        MODEL_RESIDUE.append(model_residues[i])
        PF_EXP.append(float(exp['PF'][i]))
        PF_EXP_ERR.append(float(exp['PF_error'][i]))
        C_SIM.append(c[i])
        C_SIM_ERR.append(c_err[i])
        HB_SIM.append(hb[i])
        HB_SIM_ERR.append(hb_err[i])

# convert to arrays
PROTEIN = np.array(PROTEIN)
MODEL_RESIDUE = np.array(MODEL_RESIDUE)
PF_EXP = np.array(PF_EXP)
PF_EXP_ERR = np.array(PF_EXP_ERR)
C_SIM = np.array(C_SIM)
C_SIM_ERR = np.array(C_SIM_ERR)
HB_SIM = np.array(HB_SIM)
HB_SIM_ERR = np.array(HB_SIM_ERR)

# calculate protection factors with all parameters
for i in range(len(beta_c_vals)):
    for j in range(len(beta_h_vals)):
        beta_c = beta_c_vals[i]
        beta_h = beta_h_vals[j]

        lnPF = beta_c * C_SIM + beta_h * HB_SIM
        lnPF_err = np.sqrt( (C_SIM_ERR * beta_c)**2 + (HB_SIM_ERR * beta_h)**2 )
        PF_SIM = np.exp(lnPF)
        PF_SIM_ERR = lnPF_err*PF_SIM

        # calculate statistics in log space
        x = np.log10(PF_EXP)
        y = np.log10(PF_SIM)
        rmse = RMSE(x,y)
        pearsonr = scipy.stats.pearsonr(x,y)[0]
        stats_matrix_rmse[i,j]=rmse
        stats_matrix_pearsonr[i,j]=pearsonr


# save stats matrix
stats_df_rmse = pd.DataFrame(stats_matrix_rmse, index=list(beta_c_vals), columns=list(beta_h_vals))
stats_df_pearsonr = pd.DataFrame(stats_matrix_pearsonr, index=list(beta_c_vals), columns=list(beta_h_vals))
stats_df_rmse.to_csv('output_results/RMSE_optimisation_matrix_BV2006_{}.csv'.format(exp_PF_dir))
stats_df_pearsonr.to_csv('output_results/Pearsonr_optimisation_matrix_BV2006_{}.csv'.format(exp_PF_dir))

# find optimal params
x_min = np.where(stats_df_rmse==np.min(stats_df_rmse))[0][0]
y_min = np.where(stats_df_rmse==np.min(stats_df_rmse))[1][0]

optimal_beta_c = beta_c_vals[x_min]
optimal_beta_h = beta_h_vals[y_min]

# calculate final PFs with optimal parameters
lnPF = optimal_beta_c * C_SIM + optimal_beta_h * HB_SIM
lnPF_err = np.sqrt( (C_SIM_ERR * optimal_beta_c)**2 + (HB_SIM_ERR * optimal_beta_h)**2 )
PF_SIM = np.exp(lnPF)
PF_SIM_ERR = lnPF_err*PF_SIM


# make final dataframe to save
df = pd.DataFrame()
df['Protein'] = PROTEIN
df['Model_residue'] = MODEL_RESIDUE
df['EXP PF'] = PF_EXP
df['EXP PF ERR'] = PF_EXP_ERR
df['SIM PF'] = PF_SIM
df['SIM PF ERR'] = PF_SIM_ERR
df.to_csv('output_results/data_EXP_vs_SIM_BV2006_optimised_betac_{:.2f}_betah_{:.2f}_{}.csv'.format(optimal_beta_c,optimal_beta_h,exp_PF_dir))

# calculate RMSE in log10(PF) and Pearson correlation coefficient
x = np.log10(df['EXP PF'])
y = np.log10(df['SIM PF'])
rmse = RMSE(x,y)
pearson_r = scipy.stats.pearsonr(x,y)[0]

print('Optimal parameteres:')
print('beta_c: {:.2f}'.format(optimal_beta_c))
print('beta_h: {:.2f}'.format(optimal_beta_h))
print('RMSE (log10 PF): {:.2f}'.format(rmse))
print('Pearson r: {:.2f}'.format(pearson_r))

