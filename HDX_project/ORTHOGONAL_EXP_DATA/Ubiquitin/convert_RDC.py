import sys

inputname = sys.argv[1]
outputname = sys.argv[2]

with open(inputname, 'r') as f:
    lines = f.readlines()

residues = []
rdc = []
for line in lines:
    residues.append(int(line.split()[0]))
    rdc.append(float(line.split()[4]))

with open(outputname, 'w') as g:
    for res, val in zip(residues, rdc):
        g.write("{}\t{}\n".format(res, val))


