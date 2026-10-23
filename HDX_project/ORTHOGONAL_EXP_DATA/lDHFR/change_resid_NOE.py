input_file = "tmpNOE.dat"
output_file = "tmpNOE_plus1.dat"

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        # skip empty or malformed lines
        if len(parts) < 6:
            continue
        try:
            # increment residue numbers in columns 0 and 2
            parts[0] = str(int(parts[0]) + 1)
            parts[2] = str(int(parts[2]) + 1)
        except ValueError:
            # skip lines that don't start with numbers
            continue

        fout.write(" ".join(parts) + "\n")

