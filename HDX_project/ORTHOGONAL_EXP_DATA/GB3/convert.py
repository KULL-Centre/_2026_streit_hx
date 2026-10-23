#!/usr/bin/env python3

"""
convert_mr_to_txt.py

Reads an input.mr-style distance restraint file and converts it to simplified output.txt format.

Rules:
- Skip lines with a distance value of 0.00.
- Convert Q-prefixed atom names (QD1, QG2, etc.) → H + same suffix + '*'
  e.g., QD1 → HD1*, QG → HG*, QE → HE*
- Keep only residue numbers and atom names.
- Append 1.00 1.00 as lower/upper bounds to every line.
"""

import re
from pathlib import Path

INPUT_FILE = "input.mr"
OUTPUT_FILE = "output.txt"

def convert_atom_name(name: str) -> str:
    """Convert atom name with Q prefix to H*, otherwise return unchanged."""
    name = name.strip()
    if name.startswith("Q"):
        return "H" + name[1:] + "*"
    return name

def parse_line(line: str):
    """Parse a valid data line and return converted output."""
    parts = line.split()
    if len(parts) < 7:
        return None

    try:
        resid1, resname1, atom1 = int(parts[0]), parts[1], parts[2]
        resid2, resname2, atom2 = int(parts[3]), parts[4], parts[5]
        dist = float(parts[6])
    except ValueError:
        return None

    if dist == 0.0:
        return None

    atom1 = convert_atom_name(atom1)
    atom2 = convert_atom_name(atom2)
    return f"{resid1} {atom1} {resid2} {atom2} {dist:.2f} 1.00 1.00"

def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"❌ Input file '{INPUT_FILE}' not found.")
        return

    lines = input_path.read_text().splitlines()
    output_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = parse_line(line)
        if parsed:
            output_lines.append(parsed)

    output_path.write_text("\n".join(output_lines) + "\n")
    print(f"✅ Wrote {len(output_lines)} restraints to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

