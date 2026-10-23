import re

input_file = "input.mr"
output_file = "output.txt"

pattern = re.compile(
    r"ASSIGN\s+\(RESID\s+(\d+)\s+AND\s+NAME\s+([A-Za-z0-9#*]+)\s*\)\s+\(RESID\s+(\d+)\s+AND\s+NAME\s+([A-Za-z0-9#*]+)\s*\)\s+([\d\.\-Ee]+)\s+([\d\.\-Ee]+)\s+([\d\.\-Ee]+)"
)

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        match = pattern.search(line)
        if match:
            resid1, name1, resid2, name2, val1, val2, val3 = match.groups()
            # Replace # with * in atom names
            name1 = name1.replace("#", "*")
            name2 = name2.replace("#", "*")
            f_out.write(f"{resid1} {name1} {resid2} {name2} {val1} {val2} {val3}\n")

print(f"Conversion complete. Output saved to '{output_file}'.")

