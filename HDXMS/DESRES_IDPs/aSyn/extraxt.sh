
for xtc in ../*.xtc; do
    base=$(basename "$xtc" .xtc)
    gmx_mpi trjconv -s protein.pdb -f "$xtc" -fr frames.ndx -o "${base}.xtc" << EOF
0
EOF
done

gmx_mpi trjcat -f *.xtc -cat -o traj_downsampled.xtc

if [ -f traj_downsampled.xtc ]; then
    rm jacs*.xtc
fi

