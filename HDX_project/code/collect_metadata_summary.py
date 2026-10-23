import numpy as np
import sys
import os
import pandas as pd
import mdtraj as mdj
import re
sys.path.insert(1, '/projects/lindorff_group/people/thd711/HDX_project/code/')

# import functions
from misc import read_data

# get a list of all data files containing measured H(D)X rates (some proteins can have 
# multiple files, i.e. different datasets/conditions/types of experiments)
datasets = []

for file in os.listdir('EXP_DATA/'):
    if file.endswith('.csv'):
        datasets.append(file)

sorted_datasets = sorted(datasets, key=str.lower)

# start writing output file
with open('EXP_DATA/summary_metadata.txt','w') as f:

    # iterate through all proteins
    for protein in sorted_datasets:

        if protein!= sorted_datasets[0]:
            f.write('\n')

        print(protein)
        data = read_data('EXP_DATA/{}'.format(protein))
            
        # extract metadata
        T = data.attrs['temp']
        pH = data.attrs['pH']
        ionic = data.attrs['ionic']
        exchange = data.attrs['type']
        conc = data.attrs['conc']*100 # convert fraction to %

        f.write('{} {} {} {} {} {}'.format(protein[:-4],T,ionic,pH,exchange,conc))
        
