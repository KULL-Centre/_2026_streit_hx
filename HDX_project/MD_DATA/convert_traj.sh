# to convert dcd to xtc files

conda activate mdtraj

for file in *.dcd; do mdconvert $file -o "${file%.*}.xtc"; done

conda deactivate





