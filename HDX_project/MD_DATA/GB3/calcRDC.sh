module load miniconda/py39_23.1
source activate EnsembleGen

export LD_LIBRARY_PATH=/home/thd711/lib32

# iterate through through all models and proteins

mkdir RDC
cd RDC
cp /projects/lindorff_group/people/thd711/software/scripts/pales_wrapper_traj_global.py ./

for traj in ../traj19.xtc; do
    name=$(basename "$traj" .xtc)
    rm -r task*
    python pales_wrapper_traj_global.py ../protein.pdb $traj --rdc ; mv task*/RDCg.csv RDCg_${name}.csv

done

rm pales_wrapper_traj_global.py
cd ..
    
echo "Done"

conda deactivate
