import mdtraj as md
import numpy as np
import sys

pdb = str(sys.argv[1])
traj = str(sys.argv[2])
save_dir = str(sys.argv[3])

try:
    save_framewise = sys.argv[4]
except:
    save_framewise = False

def compute_water_hbonds(trj, threshold_hbond = 0.20, treshold_angle = 150.0, save_framewise = save_framewise ):
    """
    Compute amide hydrogen to water hydrogen bonds (distance and angular cutoff).
    This version also includes a hydrogen bond angle threshold to count hbonds (default 150 degrees or more like GROMACS hbond).
    We assume the mdtraj trajectory to contain the protein in chain 0 and no solvent. If there are multiple chains
    we assume them to be identical (i.e., dimers). The script will not work correctly if these assumptions are not met!
    """
    
    
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices and residue numbers
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]
    
    hbonds_res = np.zeros((len(trj), len(amide_res_idx)))

    # water selection
    water_idx = trj.top.select("resname HOH and type O")
    
    for i, resid in enumerate(amide_res_idx):
    
        amideN_idx = trj.top.select(f"protein and name N and resid {resid}")
        amideH_idx = trj.top.select(f"protein and (name H or name HN) and resid {resid}")
        neigh = md.compute_neighbors(trj, threshold_hbond, amideH_idx, water_idx)

        # check D-H...A angles        
        hbonds_count = []
        for frame, frame_neighbors in enumerate(neigh):
            count = 0
            frame_xyz = trj.xyz[frame]  # only load coordinates for this frame

            for h_idx, donor_idx in zip(amideH_idx, amideN_idx):
                vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                vec_dh = vec_dh / np.linalg.norm(vec_dh)
        
                for a_idx in frame_neighbors:
                    vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                    vec_ha = vec_ha / np.linalg.norm(vec_ha)
                    
                    angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                    if angle_rad >= np.deg2rad(treshold_angle):
                        count += 1
            hbonds_count.append(count)
        hbonds_res[:, i] = hbonds_count

    # fraction open (at least 1 hydrogen bond to water)
    bool_open = (hbonds_res>=1)*1
    fopen = np.mean(bool_open, axis=0)

    if save_framewise=='True':
        np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
        np.savetxt(save_dir+'amideresidues.txt',amide_resid)
        np.savetxt(save_dir+'fraction_open_wathbond_{}.txt'.format(traj[:-4]),fopen)
        np.save(save_dir+'amideH_water_hbonds_frames_{}.npy'.format(traj[:-4]),hbonds_res)
        
    else:
        np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
        np.savetxt(save_dir+'amideresidues.txt',amide_resid)
        np.savetxt(save_dir+'fraction_open_wathbond_{}.txt'.format(traj[:-4]),fopen)

def main():
    trj = md.load(traj,top=pdb)
    compute_water_hbonds(trj, save_framewise = save_framewise)

if __name__ == "__main__":
    main()
