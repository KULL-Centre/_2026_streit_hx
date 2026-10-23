import numpy as np
import sys
import os
from os import listdir
import pandas as pd
import mdtraj as mdj
import scipy.stats
from scipy.optimize import curve_fit
import re
sys.path.insert(1, '/projects/lindorff_group/people/thd711/HDX_project/code/')

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

def fun_M8(X,beta_s,gamma_s,beta_p,gamma_p):

    sasa, mindist = X
    return beta_s * ( sasa ** -gamma_s) + beta_p * (mindist ** -gamma_p)

# read list of proteins to calculate PFs from a text file
input_proteins_file = sys.argv[1]
with open(input_proteins_file,'r') as f:
    lines = f.readlines()

proteins = []
for line in lines:
    proteins.append(line.strip())

# forward model parameters dictionary
beta_s_vals = np.linspace(0,1,100)
beta_p_vals = np.linspace(0,5,100)

# matrices with statistics
stats_matrix_rmse = np.zeros((100,100))
stats_matrix_pearsonr = np.zeros((100,100))

# setup lists for dataframe with data
PROTEIN = []
MODEL_RESIDUE = []
PF_EXP = []
PF_EXP_ERR = []
D_SIM = []
D_SIM_ERR = []
SA_SIM = []
SA_SIM_ERR = []

# match calculated and experimental values/residues
for protein in proteins:
    print(protein)
    # experimental data
    exp = pd.read_csv('EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protein),index_col=0)
    model_residues = np.array(exp['Model_residue']).astype(int)
    
    # model pdb
    u = mdj.load('MD_DATA/{}/protein.pdb'.format(protein),top='MD_DATA/{}/protein.pdb'.format(protein))
    amides = u.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    amides_resnames = np.array([u.top.atom(i).residue.name for i in amides])
    amides_resids = np.array([u.top.atom(i).residue.resSeq for i in amides])

    # load amide res idxs
    amide_res_idx = np.loadtxt('MD_DATA/{}/M8M9/amideresidx.txt'.format(protein))
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
    tmpsa = []
    tmpd = []
    for traj in trajs:
        tmpsa.append(np.loadtxt('MD_DATA/{}/M8M9/sasa_{}.txt'.format(protein,traj[:-4])))
        tmpd.append(np.loadtxt('MD_DATA/{}/M8M9/mindist_{}.txt'.format(protein,traj[:-4])))
    tmpsa = np.array(tmpsa)
    tmpd = np.array(tmpd)

    sasa = np.mean(tmpsa,axis=0)
    mindist = np.mean(tmpd,axis=0)

    # monomer vs dimer
    if n_prot==1: # monomer
        sasa = sasa[exp_idxs]
        mindist = mindist[exp_idxs]

    elif n_prot==2: # dimer - average the chains
        exp_idxs_1 = exp_idxs[:,0]
        exp_idxs_2 = exp_idxs[:,1]
        assert exp_idxs_1.shape[0]==exp_idxs_2.shape[0]

        sasa1 = sasa[exp_idxs_1]
        mindist1 = mindist[exp_idxs_1]

        sasa2 = sasa[exp_idxs_2]
        mindist2 = mindist[exp_idxs_2]

        # average the chains
        sasa = sasa1*0.5+sasa2*0.5
        mindist = mindist1*0.5+mindist2*0.5

    else:
        break
        print('Error: more than two protein copies are not yet supported')

    # append all data to lists
    for i in range(len(model_residues)):
        PROTEIN.append(protein)
        MODEL_RESIDUE.append(model_residues[i])
        PF_EXP.append(float(exp['PF'][i]))
        PF_EXP_ERR.append(float(exp['PF_error'][i]))
        D_SIM.append(mindist[i])
        SA_SIM.append(sasa[i])

# convert to arrays
PROTEIN = np.array(PROTEIN)
MODEL_RESIDUE = np.array(MODEL_RESIDUE)
PF_EXP = np.array(PF_EXP)
PF_EXP_ERR = np.array(PF_EXP_ERR)
D_SIM = np.array(D_SIM)
SA_SIM = np.array(SA_SIM)

# fit forward model

# initial guess
p0 = 1.3e-3, 2.64, 3.65e1, 1.27
popt, pcov = curve_fit(fun_M8, (SA_SIM,D_SIM), np.array(np.log(PF_EXP)), p0)
print('Params: ', popt)
print('Errors: ', np.sqrt(np.diag(pcov)))
beta_s, gamma_s, beta_p, gamma_p = popt

print('Fit M8 model parameters:')
print('beta_s: {:.3e}'.format(beta_s))
print('gamma_s: {:.3e}'.format(gamma_s))
print('beta_p: {:.3e}'.format(beta_p))
print('gamma_p: {:.3e}'.format(gamma_p))

# setup lists for final dataframe with optimal parameters and calculated PFs
PROTEIN = []
MODEL_RESIDUE = []
PF_EXP = []
PF_EXP_ERR = []
PF_SIM = []
PF_SIM_ERR = []

# match calculated and experimental values/residues
for protein in proteins:
    print(protein)
    # experimental data
    exp = pd.read_csv('EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protein),index_col=0)
    model_residues = np.array(exp['Model_residue']).astype(int)
    
    # model pdb
    u = mdj.load('MD_DATA/{}/protein.pdb'.format(protein),top='MD_DATA/{}/protein.pdb'.format(protein))
    amides = u.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    amides_resnames = np.array([u.top.atom(i).residue.name for i in amides])
    amides_resids = np.array([u.top.atom(i).residue.resSeq for i in amides])

    # load amide res idxs
    amide_res_idx = np.loadtxt('MD_DATA/{}/M8M9/amideresidx.txt'.format(protein))
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
    tmpPF = []
    for traj in trajs:
        sasa = np.loadtxt('MD_DATA/{}/M8M9/sasa_{}.txt'.format(protein,traj[:-4]))
        mindist = np.loadtxt('MD_DATA/{}/M8M9/mindist_{}.txt'.format(protein,traj[:-4]))
        lnPF = beta_s * (sasa ** -gamma_s) + beta_p * (mindist ** -gamma_p)
        pf = np.exp(lnPF)
        tmpPF.append(pf)

    tmpPF = np.array(tmpPF)
    PF = np.mean(tmpPF,axis=0)
    PF_err = np.std(tmpPF,axis=0)/np.sqrt(n_trajs)


    # monomer vs dimer
    if n_prot==1: # monomer
        PF = PF[exp_idxs]
        PF_err = PF_err[exp_idxs]

    elif n_prot==2: # dimer - average the chains
        exp_idxs_1 = exp_idxs[:,0]
        exp_idxs_2 = exp_idxs[:,1]
        assert exp_idxs_1.shape[0]==exp_idxs_2.shape[0]

        PF1 = PF[exp_idxs_1]
        PF_err1 = PF_err[exp_idxs_1]

        PF2 = PF[exp_idxs_2]
        PF_err2 = PF_err[exp_idxs_2]

        # average the chains
        PF = PF1*0.5+PF2*0.5
        PF_err = np.sqrt(PF_err1**2+PF_err2**2)/2

    else:
        break
        print('Error: more than two protein copies are not yet supported')

    # append all data to lists
    for i in range(len(model_residues)):
        PROTEIN.append(protein)
        MODEL_RESIDUE.append(model_residues[i])
        PF_EXP.append(float(exp['PF'][i]))
        PF_EXP_ERR.append(float(exp['PF_error'][i]))
        PF_SIM.append(PF[i])
        PF_SIM_ERR.append(PF_err[i])

# make final dataframe to save
df = pd.DataFrame()
df['Protein'] = PROTEIN
df['Model_residue'] = MODEL_RESIDUE
df['EXP PF'] = PF_EXP
df['EXP PF ERR'] = PF_EXP_ERR
df['SIM PF'] = PF_SIM
df['SIM PF ERR'] = PF_SIM_ERR
df.to_csv('output_results/data_EXP_vs_SIM_M8preavg_optimised.csv')

# calculate RMSE in log10(PF) and Pearson correlation coefficient
x = np.log10(df['EXP PF'])
y = np.log10(df['SIM PF'])
rmse = RMSE(x,y)
pearson_r = scipy.stats.pearsonr(x,y)[0]

print('Model performance:')
print('RMSE (log10 PF): {:.2f}'.format(rmse))
print('Pearson r: {:.2f}'.format(pearson_r))

