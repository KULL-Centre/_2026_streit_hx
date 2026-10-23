import sys
scripts_dir = '/projects/lindorff_group/people/thd711/software/scripts/'
sys.path.append(scripts_dir)

import os
import json
import numpy as np
import pandas as pd
import mdtraj as md
import potenci

outpath = './potenci_shifts/'

for dirname in os.listdir('.'):

    cs_path = os.path.join(dirname, 'CS.dat')
    if os.path.isfile(cs_path):
        print(dirname)
        # load experimental conditions table
        df = pd.read_csv(dirname+'/info.csv')
        df = df[df['Experiment']=='CS']
        temp = list(df['Temp(K)'])[0]
        pH = list(df['pH'])[0]
        ionic = list(df['Ionic(M)'])[0]

        # load sequence
        with open(dirname+'/seq.fasta','r') as f:
            seq= str(f.readlines()[1].strip())

        # run potenci prediction
        potenci.main(seq=seq, pH=pH, temperature=temp, ion=ionic, pkacsvfile=False, outfile=outpath+str(f'{dirname}_RCshifts.dat'))
    else:
        continue

print('Done')


