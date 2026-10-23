import sys

def increment_residue_numbers(input_pdb, output_pdb):
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM","TER")):
                # Extract and increment residue number (columns 23–26)
                resnum_str = line[22:26]
                try:
                    resnum = int(resnum_str)
                    new_resnum = resnum + 1
                    new_line = line[:22] + f"{new_resnum:4d}" + line[26:]
                    f_out.write(new_line)
                except ValueError:
                    # Handle lines where residue number isn't an integer
                    f_out.write(line)
            else:
                f_out.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_resids.py input.pdb output.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]
    increment_residue_numbers(input_pdb, output_pdb)
    print(f"Residue numbers incremented and saved to {output_pdb}")


