import re

input_file = "input.mr"
output_file = "output.txt"

# mapping for ambiguous proton groups → wildcard notation
wildcard_map = {
    "QA": "HA*",
    "QB": "HB*",
    "QG": "HG*",
    "QG1": "HG1*",
    "QG2": "HG2*",
    "QD": "HD*",
    "QD1": "HD1*",
    "QD2": "HD2*",
    "QE": "HE*",
    "QZ": "HZ*",
    "QH": "HH*",
    "QQ": "HQ*",
}

# adjust atom naming
# HX2 --> HX1, HX3 --> HX2 where HX is HA, HB or HG
hname_map = {"HA2":"HA1", "HA3":"HA2", "HB2":"HB1", "HB3":"HB2", "HG2":"HG1", "HG3":"HG2", "HD2":"HD1", "HD3":"HD2"}

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # match a typical line like:  89 GLU  HB3    90 ILE  H       4.57 #SUP  7.06
        match = re.match(
            r"^\s*(\d+)\s+\S+\s+(\S+)\s+(\d+)\s+\S+\s+(\S+)\s+([\d.]+)", line
        )
        if not match:
            continue

        res1, atom1, res2, atom2, dist = match.groups()
        # skip any line where either atom starts with QQ
        if atom1.upper().startswith("QQ") or atom2.upper().startswith("QQ"):
            continue
        # apply wildcard conversions
        atom1 = wildcard_map.get(atom1.upper(), atom1.upper())
        atom2 = wildcard_map.get(atom2.upper(), atom2.upper())

        # apply hname conversions
        atom1 = hname_map.get(atom1.upper(), atom1.upper())
        atom2 = hname_map.get(atom2.upper(), atom2.upper())

        # write converted format with 1.0 1.0 uncertainties
        fout.write(f"{res1} {atom1} {res2} {atom2} {dist} 1.0 1.0\n")

