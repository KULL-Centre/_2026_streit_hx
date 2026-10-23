import numpy as np
import mdtraj as mdj
from simtk import unit, openmm
from simtk.openmm.app import PDBFile, ForceField, Modeller
import sys
import json
import argparse

from openmm.app import PDBFile, Modeller, ForceField, NoCutoff
from openmm import unit
from collections import defaultdict

"""
Compute protection factors from the local electrostatic potential calculated on the amide hydrogen atoms
and using atomic partial charges (charges from residue of interest and preceding residues are excluded
as these factors are already corrected experimentally with a set of kint values). 
We assume the mdtraj trajectory to contain the protein in chain 0 and no solvent. If there are multiple chains
we assume them to be identical (i.e., dimers). The script will not work correctly if these assumptions are not met (e.g., heterodimers)!
This analysis also assumes HX to be catalysed primarily by hydroxide, i.e., pH > 4.0.
Structural ions like Mg2+ or Ca2+ are included in the analysis.
"""

parser = argparse.ArgumentParser(description="Process MD simulation input arguments.")

parser.add_argument("--pdb", type=str, help="Path to the PDB file.")
parser.add_argument("--topfile", type=str, help="Path to the topology file.")
parser.add_argument("--traj", type=str, help="Path to the trajectory file.")
parser.add_argument("--json_name", type=str, default='partial_charges_ff14SB.json', help="File name of the json file with partial charges. Default: partial_charges_ff14SB.json")
parser.add_argument("--charge_scaling", type=float, default=1.0, help="Scaling factor for the charges. Default: 1.0 (no scaling)")
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
json_name = args.json_name
charge_scaling = args.charge_scaling

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

def calculate_debye_huckel_potential(positions, amideH_indices, residues, chains, charges, debye_len, temperature=298.15):
    """
    Calculate Debye-Hückel potential at each amide hydrogen atom.
    By default, all charges from the same amino acid and preceding residues are excluded here (because intrinsic HDX rates have been corrected for neighbour effects).
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
        bool_res = ((residues != residue_i) & (residues != residue_i - 1)) * 1 # exclude same residue and previous residues
        bool_chain = ( np.abs(chain_i - chains) > 0) * 1 # keep if residue in a different chain
        bool = bool_res + bool_chain # i.e., both the residue and chain criteria. 0 = exclude; 1,2 = keep
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

def get_charge_dict(json_file):
    """Return (residue_index, atom_name) list to uniquely identify atoms."""

    # Load the JSON file as a dictionary
    with open(json_file, 'r') as f:
        charge_dict = json.load(f)

    return charge_dict

CHARMM_NTER_CHARGES = {
    "N": -0.30,
    "HT1": 0.33,
    "HT2": 0.33,
    "HT3": 0.33,
    "CA": 0.21,
    "HA": 0.10,
}

CHARMM_CTER_CHARGES = {
    "C": 0.34,
    "OT1": -0.67,
    "OT2": -0.67,
}

# Useful aliases depending on the PDB/topology atom names
CHARMM_NTER_ALIASES = {
    "H":  "HT1",
    "H1": "HT1",
    "H2": "HT2",
    "H3": "HT3",
}

CHARMM_CTER_ALIASES = {
    "O": "OT1",
    "OXT": "OT2",
}


def lookup_charge(resname, atom, charge_dict):
    """
    Works for:
    - AMBER-style JSONs with NALA/CALA etc.
    - CHARMM-style JSONs with normal ALA plus NTER/CTER patches not present in JSON.
    """

    # First try direct lookup.
    # This preserves your current AMBER behaviour.
    if resname in charge_dict:
        atom_lookup = atom

        # Your existing AMBER N-terminal H rename
        if resname[0] == "N" and atom_lookup == "H":
            atom_lookup = "H1"

        if atom_lookup in charge_dict[resname]:
            return float(charge_dict[resname][atom_lookup])

    # If direct lookup failed, try CHARMM-style terminal patch logic.
    is_nterm = resname.startswith("N") and resname[1:] in charge_dict
    is_cterm = resname.startswith("C") and resname[1:] in charge_dict

    if is_nterm:
        base_resname = resname[1:]
        atom_lookup = CHARMM_NTER_ALIASES.get(atom, atom)

        # Atoms changed/added by NTER patch
        if atom_lookup in CHARMM_NTER_CHARGES:
            return CHARMM_NTER_CHARGES[atom_lookup]

        # HN is removed by NTER, so it should not normally appear
        if atom_lookup == "HN":
            raise KeyError(f"{resname} atom {atom}: HN should be removed by CHARMM NTER patch")

        # Otherwise use normal residue charge
        return float(charge_dict[base_resname][atom_lookup])

    if is_cterm:
        base_resname = resname[1:]
        atom_lookup = CHARMM_CTER_ALIASES.get(atom, atom)

        # Atoms changed/added by CTER patch
        if atom_lookup in CHARMM_CTER_CHARGES:
            return CHARMM_CTER_CHARGES[atom_lookup]

        # O is removed by CTER and replaced by OT1/OT2.
        # If atom name was O, alias above maps it to OT1.
        return float(charge_dict[base_resname][atom_lookup])

    # Histidine fallback, keeping your existing logic
    if resname == "HIS":
        try:
            return float(charge_dict["HID"][atom])
        except KeyError:
            return float(charge_dict["HIE"][atom])

    raise KeyError(f"Could not find charge for residue {resname}, atom {atom}")

def get_charges(pdb_filename,json_file='/projects/lindorff_group/people/thd711/HDX_project/code/partial_charges_ff14SB.json', charge_scaling=charge_scaling):

    # get charge dictionary
    charge_dict = get_charge_dict(json_file)

    # Load original structure and get residue/atom information
    # Load the PDB file
    pdb = PDBFile(pdb_filename)

    # Get the topology
    topology = pdb.topology

    # List to store atom information
    residue_numbers = []
    residue_names = []
    atom_names = []

    # Iterate through all atoms in the topology
    for atom in topology.atoms():
        residue = atom.residue
        residue_numbers.append(int(residue.id)) # Actual residue number from PDB file
        residue_names.append(residue.name) # Residue name (e.g., ALA, GLY)
        atom_names.append(atom.name) # Atom name (e.g., CA, N)
        
    # adjust names of terminal residues with "N" and "C"
    final_residue_names = []
    start_residue = np.min(residue_numbers)
    end_residue = np.max(residue_numbers)

    for resname,resid in zip(residue_names,residue_numbers):
        if resid==start_residue:
            final_residue_names.append(str('N')+resname)
        elif resid==end_residue:
            final_residue_names.append(str('C')+resname)
        else:
            final_residue_names.append(resname)

    # list of partial charges
    charges = []
    for resname,atom,resnum in zip(final_residue_names,atom_names,residue_numbers):

        ch = lookup_charge(resname, atom, charge_dict)
        
        charges.append(ch)

    return np.array(charges) * charge_scaling


def phi_to_lnPF(potential, temperature = 298.15):
    
    # Calculate the PF from the electrostatic potential given in Volts

    return ( (z * F * potential) / ( R * temperature ) )


########################
### RUN THE ANALYSIS ###
########################
    
# load trajectory
trj = mdj.load(traj,top=pdb)
CAind = trj.top.select('name CA and not (resname PRO or resname NH2 or resname ACE or resname NME)')
# Get the residue indices
amide_resid = [trj.top.atom(i).residue.resSeq for i in CAind]
amide_res_idx = [trj.top.atom(i).residue.index for i in CAind]

# check how many protein chains and if the chains are identical
nprot_chains = trj.atom_slice(trj.top.select('name CA')).n_chains

# obtain the charges and convert to Coulomb
charges = get_charges(topfile, '/projects/lindorff_group/people/thd711/HDX_project/code/'+json_name, charge_scaling=charge_scaling)
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

    phi = calculate_debye_huckel_potential(positions, amideH_indices, residues, chainids, charges, debye_len, temperature = temperature)

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

