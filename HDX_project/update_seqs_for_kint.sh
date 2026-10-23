#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cd MD_DATA

for dir in *; do cp $dir/structure_for_seq.pdb ../EXP_DATA/sequences_for_kint/$dir.pdb; done

cd ../EXP_DATA/sequences_for_kint/

cp $code_path/get_seq_from_pdb.py ./

for file in *.pdb; do python get_seq_from_pdb.py $file; done

cd ../../

echo "Sequences from PDB files updated!"

conda deactivate

