import xml.etree.ElementTree as ET
import json

# Path to your ff14SB XML file
input_xml = 'charmm36.xml'
output_json = 'partial_charges_charmm36.json'

# Parse the XML file
tree = ET.parse(input_xml)
root = tree.getroot()

# Find the Residues section
residues_section = root.find(".//Residues")

# Create dictionary to hold partial charges
charge_dict = {}

# Iterate over all residues and extract atom charges
for residue in residues_section.findall("Residue"):
    res_name = residue.attrib['name']
    charge_dict[res_name] = {}

    for atom in residue.findall("Atom"):
        atom_name = atom.attrib['name']
        charge = float(atom.attrib['charge'])
        charge_dict[res_name][atom_name] = charge

# Save the dictionary to a JSON file
with open(output_json, 'w') as f:
    json.dump(charge_dict, f, indent=2)

print(f"Partial charges saved to {output_json}")
