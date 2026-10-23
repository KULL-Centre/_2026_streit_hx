#!/bin/bash

export LD_LIBRARY_PATH=/opt/software/python/3.9.9/lib:/opt/software/gcc/11.2.0/lib64:/opt/software/gcc/11.2.0/lib:/opt/software/openmpi/4.1.0/lib:/opt/software/openblas/0.3.24/lib:/opt/software/cuda/11.8/lib64:/opt/software/cuda/11.8/extras/Debugger/lib64:/opt/software/cuda/11.8/extras/CUPTI/lib64:/opt/software/fftw/3.6.9/lib

module load fftw/3.6.9
module load cuda/11.8
module load openblas
module load openmpi/4.1.0
module load gcc/11.2.0
module load python/3.9.9
module load gromacs/2024

save_dir=rmsd_data_h4/

for DIR in {Im7,}; do
    DIR="${DIR%/}"
    cd "$DIR"

    if compgen -G "${save_dir}*.xvg" > /dev/null; then
        echo "Skipping $DIR: $save_dir already contains .xvg files"
        cd ..
        continue
    fi

    mkdir -p "$save_dir"

    for traj in *.xtc; do
        gmx_mpi rms -s protein.pdb -f "$traj" -n index.ndx -o "${save_dir}rmsd_${traj}.xvg" << EOF
BB
h4
EOF
    done

    cd ..
    echo "Done with $DIR!"
done

conda deactivate
