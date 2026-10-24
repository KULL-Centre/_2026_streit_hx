
cd ../DESRES-Trajectory_pnas2018b-NTail-a99SBdisp-protein/pnas2018b-NTail-a99SBdisp-protein/

for file in *.dcd; do mdconvert $file -o "../../converted_XTC/${file%.*}.xtc"; done

cd ../../converted_XTC/






