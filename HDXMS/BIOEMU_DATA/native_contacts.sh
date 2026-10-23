#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDXMS/code/'
save_dir='native_contacts/'


for DIR in {PROTEIN,}; do
        
        DIR="${DIR%/}"   # Remove trailing slash
        echo $DIR
        cd "$DIR"

        mkdir -p $save_dir

        for traj in *.xtc; do
                python "$code_path/calc_Q.py" \
                        --pdb topology.pdb \
                        --xtc $traj \
                        --contacts "ref_structure/contacts.txt" \
                        --output "$save_dir/Q_$traj.txt"
        done

        cd ..
done


conda deactivate


