module load miniconda/py39_23.1
source activate HDXproject

export LD_LIBRARY_PATH=/home/thd711/lib32

mkdir RDC
cd RDC
cp /projects/lindorff_group/people/thd711/software/scripts/calc_exp_data_localpales.py ./

# iterate through through all models and proteins
for traj in ../*.xtc; do
    name=$(basename "$traj" .pdb)
    rm -r task*
    python calc_exp_data_localpales.py ../protein.pdb $traj --rdc ; mv task*/RDC.csv RDC_${name}.csv
    rm -r task*
done
rm calc_exp_data_localpales.py

cd ..
    
echo "Done"

conda deactivate
