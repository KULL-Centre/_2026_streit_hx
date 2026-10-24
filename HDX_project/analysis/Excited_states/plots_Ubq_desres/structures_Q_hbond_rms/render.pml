# -------------------------
# 0. Load all selected PDBs and align globally
# -------------------------

reinitialize

python
from pymol import cmd
import glob
import os

pdb_files = sorted(glob.glob("*.pdb"))

if len(pdb_files) == 0:
    raise RuntimeError("No PDB files found. Run this script from inside selected_pdbs/")

objs = []

for pdb in pdb_files:
    # make PyMOL-safe object name from filename
    obj = os.path.splitext(os.path.basename(pdb))[0]
    obj = obj.replace(".", "_").replace("-", "_")

    cmd.load(pdb, obj)
    objs.append(obj)

print(f"Loaded {len(objs)} PDB files")

ref_obj = objs[0]
print(f"Reference object: {ref_obj}")

# assign secondary structure, useful if the PDBs do not contain HELIX/SHEET records
cmd.dss("all")

# align all structures to reference using C-alpha atoms
for obj in objs:
    if obj == ref_obj:
        continue

    mob_sel = f"{obj} and polymer.protein and name CA"
    ref_sel = f"{ref_obj} and polymer.protein and name CA"

    try:
        rmsd = cmd.super(mob_sel, ref_sel)
        print(f"Aligned {obj} to {ref_obj}: RMSD = {rmsd[0]:.3f} Å over {int(rmsd[1])} atoms")
    except Exception as e:
        print(f"Could not align {obj}: {e}")

python end


# -------------------------
# 1. Basic display
# -------------------------
bg_color white
hide everything, all
show cartoon, polymer.protein
cartoon automatic, all
cartoon fancy_helices, on
cartoon high_quality, on
cartoon smooth_loops, on


# -------------------------
# 2. Color secondary structure
# -------------------------
# Helices: purple
color purple, ss h

# Sheets: yellow
color yellow, ss s

# Loops/coils: gray
color gray70, polymer.protein and not ss h+s

# Optional: set cartoon thickness
set cartoon_loop_radius, 0.3
set cartoon_ring_fancy, 1
set cartoon_tube_radius, 0.25


# -------------------------
# 3. Add lighting & effects
# -------------------------
set ray_trace_mode, 1
set ray_trace_gain, 0.1
set ambient, 0.5
set ray_shadows, on
set spec_count, 1
set depth_cue, 0


# -------------------------
# 4. Show additional details optional
# -------------------------
show sticks, resn ATP+ADP+HET
set stick_radius, 0.15
set stick_color, orange


# -------------------------
# 5. Camera and quality
# -------------------------
set ray_trace_fog, 0
set cartoon_transparency, 0
set cartoon_highlight_color, black

orient all
zoom all, 5


# -------------------------
# 6. Ray trace
# -------------------------
ray 2000, 2000
png protein_render.png, dpi=300
