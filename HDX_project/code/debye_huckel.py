import numpy as np
import mdtraj as mdj
from simtk import unit, openmm
from simtk.openmm.app import PDBFile, ForceField, Modeller
import argparse

"""
Compute protection factors from the local electrostatic potential calculated on the amide hydrogen atoms
and using atomic partial charges (charges from residue of interest and neighbouring residues) are excluded
as these factors are already corrected experimentally with a set of kint values. 
We assume the mdtraj trajectory to contain the protein in chain 0 and no solvent. If there are multiple chains
we assume them to be identical (i.e., dimers). The script will not work correctly if these assumptions are not met (e.g., heterodimers)!
This analysis also assumes HX to be catalysed primarily by hydroxide, i.e., pH > 4.0.
Structural ions like Mg2+ or Ca2+ are included in the analysis.
"""

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--topfile", type=str, help="Path to the topology file.")
parser.add_argument("--traj", type=str, help="Path to the trajectory file.")
parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in Kelvin. Default: 300.0")
parser.add_argument("--ionic", type=float, default=0.15, help="Ionic concentration. Default: 0.15")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs. Default: outputs")
parser.add_argument("--stride", type=int, default=1, help="Stride for analysing trajectory frames. Default: 1.")
parser.add_argument("--save_frames", type=str, default="False", help="Whether to save data for all frames in a numpy array. Default: False.")

args = parser.parse_args()

pdb = args.pdb
topfile = args.topfile
traj = args.traj
temperature = args.temperature
ionic = args.ionic
save_dir = args.save_dir
stride = args.stride
save_framewise = args.save_frames
    
# Constants
EPSILON_0 = 8.854187817e-12 # Vacuum permittivity (F/m or C^2/(N m^2))
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
NA = 6.02214076e23          # Avogadro's number
kB = 1.380649e-23           # J/K (Boltzmann constant)
F = 9.648533e4              # Faraday constant, C/mol
z = -1.0                    # charge of hydroxide ion
R = 8.314463                # Gas constant, J K-1 mol-1

def debye_length(ionic_strength, temperature=298.15):

    # ionic strength input in units of M (molar)

    # conversions
    Im = ionic_strength * 1e3  # mol/m^3

    # dielectric
    # temperature dependence of dielectric constant using the empirical formula proposed by Akerlof & Oshry (1950)
    dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
    
    numerator = dielectric * EPSILON_0 * kB * temperature
    denominator = 2 * NA * (E_CHARGE ** 2) * Im
    
    lambda_D = np.sqrt(numerator / denominator)  # in meters
    
    return lambda_D

def calculate_debye_huckel_potential(positions, amideH_indices, residues, chains, charges, debye_len, exclude = 1, temperature=298.15):
    """
    Calculate Debye-Hückel potential at each amide hydrogen atom.
    By default, all charges from the same amino acid and neighbouring residues are excluded here (because intrinsic HDX rates have been corrected for neighbour effects).
    This is defined by the "exclude" parameter.
    
    Parameters:
    - positions: Nx3 array in nanometers
    - charges: length-N array in elementary charges
    - debye_len: Debye length in meters
    - dielectric: relative permittivity
    
    Returns:
    - potentials: length-N array of potentials in Volts
    """
    N = len(positions)
    potentials = []

    # calculate dielectric
    # temperature dependence of dielectric constant using the empirical formula proposed by Akerlof & Oshry (1950)
    dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
    
    for idx in amideH_indices:
        r_i = np.array([positions[idx]])
        residue_i = residues[idx]
        chain_i = chains[idx]

        # boolean matrix whether or not to exclude the pair-wise contributions from the same or neighbouring residues
        # 1 = used
        # 0 = excluded
        bool_res = ( np.abs(residue_i - residues) > exclude) * 1 # keep if separated more by more than number of exclusion residues
        bool_chain = ( np.abs(chain_i - chains) > 0) * 1 # keep if residue in a different chain
        bool = bool_res + bool_chain # i.e., both the residue and chain criteria. 0 = exlude; 1,2 = keep
        keepidxs = np.where(bool > 0)[0]
    
        # calculate distances in nanometres
        r = np.linalg.norm( r_i - positions , axis=1)
        
        # convert to metres
        r = r[keepidxs] * 1e-9

        # extract final charges
        q = charges[keepidxs]

        # calculate potential
        pot_i = np.sum( q * np.exp(-r / debye_len) / r)

        # include physical constants
        pot_i = pot_i * ( 1 / (4 * np.pi * EPSILON_0 * dielectric ) )

        potentials.append(pot_i)
    
    # return potentials
    return np.array(potentials)


# function to obtain atomic partial charges from a force field
# this function expects the input structures to be protonated
def get_charges(pdb_file, forcefield='amber14'):
    
    # Load structure
    pdb = PDBFile(pdb_file)
    
    # Load force field
    ff = ForceField('{}-all.xml'.format(forcefield), '{}/tip3p.xml'.format(forcefield))
    
    # Create Modeller for adding hydrogens if needed
    modeller = Modeller(pdb.topology, pdb.positions)
    
    # Create system to assign forces and charges
    system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff)
    
    # Extract partial charges from system
    charges = []
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nbforce = force
            break
    
    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nbforce.getParticleParameters(i)
        charges.append(charge._value)  # charges are in elementary charge units
    
    charges = np.array(charges)

    return charges


def phi_to_lnPF(potential, temperature = 298.15):
    
    # Calculate the PF from the electrostatic potential given in Volts

    return ( (z * F * potential) / ( R * temperature ) )


########################
### RUN THE ANALYSIS ###
########################
    
# load trajectory
trj = mdj.load(traj,top=pdb)
if stride>1:
    trj = trj[::stride]
CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
# Get the residue indices
amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

# check how many protein chains and if the chains are identical
nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains

# obtain the charges and convert to Coulomb
charges = get_charges(topfile)
charges = charges * E_CHARGE

# obtain residues and chain ids for all atoms
all = trj.top.select("protein or resname NH2 or resname ACE or resname NME or type Ca or type Mg")
residues = np.array([trj.top.atom(i).residue.resSeq for i in all]).astype(int)
chainids = []

if nprot_chains==1:
    
    for res in residues:
        chainids.append(0)

elif nprot_chains==2: # if two chains

    for i in range(len(residues)):
        if i < int(len(residues)/2):
            chainids.append(0)
        else:
            chainids.append(1)

else:
    print('Error: more than two chains are not supported')

chainids = np.array(chainids).astype(int)

# get amide hydrogen indices
amideH_indices = trj.top.select('(name H or name HN) and protein')

# get debye_length / salt screening parameter
debye_len = debye_length(ionic_strength = ionic, temperature = temperature)

# list for frame-wise potentials
phis = []

for frame in range(len(trj)):

    # get positions
    positions = trj.xyz[frame][all] # all selected indices belong to protein etc

    phi = calculate_debye_huckel_potential(positions, amideH_indices, residues, chainids, charges, debye_len, exclude = 1, temperature = temperature)

    phis.append(phi)

# convert to array and PFs
phis = np.array(phis)

# average
mu_phis = np.mean(phis,axis=0)
lnPF = phi_to_lnPF(mu_phis, temperature = temperature)

# frame wise data
if save_framewise=='True':
    lnPF_frames = phi_to_lnPF(phis, temperature = temperature)
    np.save(save_dir+'lnPF_frames_{}.npy'.format(traj[:-4]),lnPF_frames)

# save data
np.savetxt(save_dir+'amideresidx.txt',amide_res_idx)
np.savetxt(save_dir+'amideresidues.txt',amide_resid)
np.savetxt(save_dir+'potentials_{}.txt'.format(traj[:-4]),mu_phis)
np.savetxt(save_dir+'lnPF_{}.txt'.format(traj[:-4]),lnPF)

