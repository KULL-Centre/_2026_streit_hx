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
foreach traj ( ../*.xtc )
    set name = `basename $traj .xtc`
    # skip if output file already exists
    if ( -e CS_${name}.csv ) then
        echo "CS_${name}.csv already exists — skipping ${name}"
        continue
    endif
    rm -rf task*
    python calc_exp_data_localpales.py ../protein.pdb $traj --cs
    mv task-0/chemical_shifts.csv CS_${name}.csv
end

# remove python script
rm calc_exp_data_localpales.py

cd ..

# deactivate conda
conda deactivate


