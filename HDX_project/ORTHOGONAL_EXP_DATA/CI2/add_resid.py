#!/usr/bin/env python3

input_file = "tmpCS.dat"
output_file = "tmpCS_modified.dat"  # change to same name if you want to overwrite

with open(input_file) as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    # Copy header or blank lines unchanged
    if line.strip() == "" or line.strip().startswith("#") or not line[0].isdigit():
        out_lines.append(line)
        continue

    parts = line.split()
    try:
        parts[0] = str(int(parts[0]) + 1)  # increment residue number
    except ValueError:
        pass  # in case the first field isn't numeric
    out_lines.append(" ".join(parts) + "\n")

with open(output_file, "w") as f:
    f.writelines(out_lines)

print(f"✅ Wrote updated file to {output_file}")

