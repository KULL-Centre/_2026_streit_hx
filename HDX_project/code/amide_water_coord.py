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

try:
    distance_cut = float(sys.argv[5])
except:
    distance_cut = 0.26

def compute_watercoord(trj, distance_threshold = distance_cut, save_framewise = save_framewise):
    """
    Compute the amide hydrogen - water oxygen coordination number according to Persson & Halle PNAS 2015 (2.6 Å distance cut-off, no angular criterion). 
    Requires an input trajectory with protein and water coordinates and PBC effects removed (e.g., gmx trjconv -pbc mol -center). 
    """
    
    CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
    # Get the residue indices and residue numbers
    amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
    amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]
    
    Hcoord = np.zeros((len(trj), len(amide_res_idx)))

    # water selection
    water_idx = trj.top.select("resname HOH and type O")

    for i, resid in enumerate(amide_res_idx):
    
        amideH_idx = trj.top.select(f"protein and (name H or name HN) and resid {resid}")
        neigh = md.compute_neighbors(trj, distance_threshold, amideH_idx, water_idx)
        Hcoord[:, i] = list(map(lambda x: len(x), neigh))
    
    # fraction open (Persson Halle)
    bool_open = (Hcoord>=2)*1
    fopen = np.mean(bool_open, axis=0)

    if save_framewise=='True':
        np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
        np.savetxt(save_dir+'amideresidues.txt',amide_resid)
        np.savetxt(save_dir+'fraction_open_PH2015_{}_{}nm.txt'.format(traj[:-4], distance_cut),fopen)
        np.save(save_dir+'amideH_water_coord_frames_{}_{}nm.npy'.format(traj[:-4], distance_cut),Hcoord)
        
    else:
        np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
        np.savetxt(save_dir+'amideresidues.txt',amide_resid)
        np.savetxt(save_dir+'fraction_open_PH2015_{}_{}nm.txt'.format(traj[:-4], distance_cut),fopen)

def main():
    trj = md.load(traj,top=pdb)
    compute_watercoord(trj, save_framewise = save_framewise)

if __name__ == "__main__":
    main()
