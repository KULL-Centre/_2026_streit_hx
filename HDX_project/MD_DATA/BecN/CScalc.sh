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
python calc_exp_data_localpales.py ../protein.pdb ../traj0.xtc --cs
mv task-0/chemical_shifts.csv CS_traj0.csv
mv task-0/Rgyr-all.csv Rgyr-all-traj0.csv
rm -rf task*

# remove python script
rm calc_exp_data_localpales.py

cd ..

# deactivate conda
conda deactivate


