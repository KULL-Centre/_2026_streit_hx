# convert_noe.py

def convert_noe_file(input_file="tmpNOE.dat", output_file="NOE.dat", offset=40):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("#"):
                # copy blank or comment lines unchanged
                outfile.write(line + "\n")
                continue

            parts = line.split()
            # Subtract 40 from the first and third numerical columns
            try:
                parts[0] = str(int(parts[0]) - offset)
                parts[2] = str(int(parts[2]) - offset)
            except ValueError:
                # if the line doesn’t have numbers where expected, copy as-is
                pass

            outfile.write(" ".join(parts) + "\n")

if __name__ == "__main__":
    convert_noe_file()
    print("✅ Conversion complete. Output written to NOE.dat")

