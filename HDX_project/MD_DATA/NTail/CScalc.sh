#!/bin/csh

module load miniconda/py39_23.1
conda activate HDXproject

# tmp dir
setenv TMPDIR /scratch/thd711/tmp

echo $TMPDIR

# Add SPARTA+ directory to the front of PATH
set path = (/projects/lindorff_group/people/thd711/software/SPARTA+ $path)

mkdir CS
cd CS

cp /projects/lindorff_group/people/thd711/software/scripts//calc_exp_data_localpales.py ./

# analyse trajectories
python calc_exp_data_localpales.py ../protein.pdb ../pnas2018b-NTail-a99SBdisp-protein-0000.xtc --cs
mv task-0/chemical_shifts.csv CS_pnas2018b-NTail-a99SBdisp-protein-0000.csv
mv task-0/Rgyr-all.csv Rgyr-all-pnas2018b-NTail-a99SBdisp-protein-0000.csv
rm -rf task*

python calc_exp_data_localpales.py ../protein.pdb ../pnas2018b-NTail-a99SBdisp-protein-0001.xtc --cs
mv task-0/chemical_shifts.csv CS_pnas2018b-NTail-a99SBdisp-protein-0001.csv
mv task-0/Rgyr-all.csv Rgyr-all-pnas2018b-NTail-a99SBdisp-protein-0001.csv
rm -rf task*

python calc_exp_data_localpales.py ../protein.pdb ../pnas2018b-NTail-a99SBdisp-protein-0002.xtc --cs
mv task-0/chemical_shifts.csv CS_pnas2018b-NTail-a99SBdisp-protein-0002.csv
mv task-0/Rgyr-all.csv Rgyr-all-pnas2018b-NTail-a99SBdisp-protein-0002.csv
rm -rf task*

# remove python script
rm calc_exp_data_localpales.py

cd ..

# deactivate conda
conda deactivate


