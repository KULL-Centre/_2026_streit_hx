#!/bin/bash

export LD_LIBRARY_PATH=/opt/software/python/3.9.9/lib:/opt/software/gcc/11.2.0/lib64:/opt/software/gcc/11.2.0/lib:/opt/software/openmpi/4.1.0/lib:/opt/software/openblas/0.3.24/lib:/opt/software/cuda/11.8/lib64:/opt/software/cuda/11.8/extras/Debugger/lib64:/opt/software/cuda/11.8/extras/CUPTI/lib64:/opt/software/fftw/3.6.9/lib

module load fftw/3.6.9
module load cuda/11.8
module load openblas
module load openmpi/4.1.0
module load gcc/11.2.0
module load python/3.9.9
module load gromacs/2024

save_dir=rmsd_data/

for DIR in */; do
        DIR="${DIR%/}"   # Remove trailing slash
	cd $DIR
	mkdir $save_dir

	for traj in *.xtc; do gmx_mpi rms -s exp_native_renum.pdb -f $traj -o $save_dir/rmsd_$traj.xvg << EOF
Backbone
Backbone
EOF
done

	cd ..
	echo "Done!"

done

conda deactivate


