import numpy as np
import sys
import os
import pandas as pd
import mdtraj as mdj
import re
sys.path.insert(1, '/projects/lindorff_group/people/thd711/HDX_project/code/')

# import functions
from misc import read_data
from kint import *

# read list of proteins to calculate PFs from a text file
input_proteins_file = sys.argv[1]
with open(input_proteins_file,'r') as f:
    lines = f.readlines()

proteins = []
for line in lines:
    proteins.append(line.strip())

# get a list of all data files containing measured H(D)X rates (some proteins can have 
# multiple files, i.e. different datasets/conditions/types of experiments)
datasets = []
for file in os.listdir('EXP_DATA/'):
    if file.endswith('.csv'):
        datasets.append(file)

# iterate through all proteins
for protein in proteins:

    print(protein)
    # find all datasets for this protein
    protein_datasets = []
    for d in datasets:
        if protein in d:
            protein_datasets.append(d)
    print(protein_datasets)

    if len(protein_datasets)==0:
        continue

    for filename in protein_datasets:
        # load data
        data = read_data('EXP_DATA/{}'.format(filename))
        
        # load sequence
        with open('EXP_DATA/sequences_for_kint/{}.txt'.format(protein),'r') as f:
            seq = f.readlines()[0].strip()

        # exception: AAQAA3 is modified due to uncertain exact experimental conditions per residue. Reported PFs are increased by a factor of 1.34896 to account for the faster intrinsic reference rates (2018 vs 1993)
        if protein=='AAQAA3':
            # load data
            data = read_data('EXP_DATA/{}'.format(filename))
            # save data
            df = pd.DataFrame()
            df['Model_residue'] = data['Model_residue']
            df['PF'] = data['PF']*1.34896
            df['PF_error'] = data['PF']*0.5
            df.to_csv('EXP_DATA/tmp_PF_Stofella_2025/{}'.format(filename))
            continue

        # extract metadata
        T = data.attrs['temp']
        pH = data.attrs['pH']
        ionic = data.attrs['ionic']
        exchange = data.attrs['type']
        conc = data.attrs['conc']*100 # convert fraction to %
        
        # get experimental rates
        residues = np.array(data['Model_residue']).astype(int)
        res_idxs = residues-1
        kobs = np.array(data['kobs'])
        kobs_error = np.array(data['kobs_error'])

        # find the first residue number (corresponding to residue index 0)
        protein_topology = mdj.load('EXP_DATA/sequences_for_kint/{}.pdb'.format(protein),'EXP_DATA/sequences_for_kint/{}.pdb'.format(protein))
        first_residue = list(protein_topology.topology.residues)[0]
        try:
            first_residue_number = int(str(first_residue)[3:])
        except:
            first_residue_number = int(str(first_residue)[4:])


        # adjust residue_idx
        res_idxs = res_idxs - (first_residue_number - 1)

        # adjust experimental rates for H2O or D2O concentration
        kobs = kobs / (conc / 100)
        kobs_error = kobs_error / (conc / 100)
        
        # calculate kint (always calculated for 100% H2O or D2O depending on the exchange type, hd_percentage
        # is only used to estimated the water pKw)
        kref = Stofella2025(sequence=seq, temperature=T, pH=pH, reference='poly', exchange_type=exchange,
                            hd_percentage=conc, ph_correction=False, wildcard='X')  
        kref = kref[res_idxs]

        # calculate PF and error
        PF = kref/kobs
        PF_error = PF*np.sqrt((kobs_error/kobs)**2)

        # save data
        df = pd.DataFrame()
        df['Model_residue'] = residues
        df['kint'] = kref
        df['PF'] = PF
        df['PF_error'] = PF_error
        df.to_csv('EXP_DATA/tmp_PF_Stofella_2025/{}'.format(filename))

    # combine multiple datasets --> merge data from mutually exclusive residue sets and average PFs where multiple PFs exist
    if len(protein_datasets) > 1:

        # load all the datasets
        DATASETS = []
        for filename in protein_datasets:
            D = pd.read_csv('EXP_DATA/tmp_PF_Stofella_2025/{}'.format(filename),index_col=0)
            DATASETS.append(D)
        
        FINAL = DATASETS[0]

        #for i in range(1,len(DATASETS)):
            #DATA = DATASETS[i]
            #FINAL = FINAL.merge(DATA, on = 'Model_residue', how = 'outer', suffixes = ('_{}'.format(i-1),'_{}'.format(i)))
        FINAL = DATASETS[0].copy()
        FINAL = FINAL.rename(columns={'kint': 'kint_0','PF': 'PF_0', 'PF_error': 'PF_error_0'})

        for i in range(1, len(DATASETS)):
            DATA = DATASETS[i].copy()
            DATA = DATA.rename(columns={'kint': f'kint_{i}','PF': f'PF_{i}', 'PF_error': f'PF_error_{i}'})
            FINAL = FINAL.merge(DATA, on='Model_residue', how='outer')

        pf_col_list = [f'PF_{i}' for i in range(len(DATASETS))]
        pferr_col_list = [f'PF_error_{i}' for i in range(len(DATASETS))]
        kint_col_list = [f'kint_{i}' for i in range(len(DATASETS))]

        #pf_col_list = []
        #for i in range(len(DATASETS)):
            #pf_col_list.append('PF_{}'.format(i))

        #pferr_col_list = []
        #for i in range(len(DATASETS)):
            #pferr_col_list.append('PF_error_{}'.format(i))

        FINAL['PF'] = FINAL[pf_col_list].mean(axis=1)
        FINAL['PF_error'] = np.array(FINAL['PF'])*0.5 # assigning 50% errors due to uncertainty in kint
        #FINAL['PF_error'] = FINAL[pf_col_list].std(axis=1)

        # replace all "NaN" in the PF_error column - i.e. residues where there is only one entry
        final_errors = []
        for i in range(len(FINAL)):
            if pd.isna(FINAL['PF_error'][i]):
                values = []
                for entry in pferr_col_list:
                    values.append(FINAL[entry][i])
                final_errors.append(next(value for value in values if not np.isnan(value)))
            else:
                final_errors.append(FINAL['PF_error'][i])

        FINAL['PF_error'] = final_errors
        
        # sort by residue
        FINAL = FINAL.sort_values(by='Model_residue', ascending=True)
        FINAL = FINAL.reset_index()
        FINAL = FINAL.drop('index',axis=1)

        FINAL.to_csv('EXP_DATA/PF_Stofella_2025/{}.csv'.format(protein))

    else: # if there is just one set

        D = pd.read_csv('EXP_DATA/tmp_PF_Stofella_2025/{}'.format(filename),index_col=0)
        D['PF_error'] = 0.5 * np.array(D['PF']) # assigning 50% errors due to uncertainty in kint
        D.to_csv('EXP_DATA/PF_Stofella_2025/{}.csv'.format(protein))










