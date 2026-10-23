#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ------------------------------------------------------------------
# 1. Read the file while preserving exact formatting information
# ------------------------------------------------------------------
def read_chemical_shift_file(filepath):
    colnames = ['RESID', 'RESNAME', 'HA', 'CA', 'HN', 'N', 'C', 'CB']
    data = []
    raw_lines = []                     # keep original non-comment lines for later reuse
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            raw_lines.append(line)

            resid = int(fields[0])
            resname = fields[1]
            values = fields[2:]
            values = [np.nan if v.lower() == 'nan' else float(v) for v in values]
            data.append([resid, resname] + values)

    df = pd.DataFrame(data, columns=colnames)
    df.set_index('RESID', inplace=True)
    return df, raw_lines


# ------------------------------------------------------------------
# 2. Isotope-specific offset calculation
# ------------------------------------------------------------------
def compute_isotope_offsets(df1, df2):
    common_res = sorted(set(df1.index) & set(df2.index))
    print(f"Common residues: {len(common_res)}\n")

    groups = {
        '1H':  {'nuclei': ['HA', 'HN'], 'v1': [], 'v2': []},
        '13C': {'nuclei': ['CA', 'CB', 'C'], 'v1': [], 'v2': []},
        '15N': {'nuclei': ['N'],           'v1': [], 'v2': []},
    }

    for resid in common_res:
        r1 = df1.loc[resid]
        r2 = df2.loc[resid]
        for iso, g in groups.items():
            for nuc in g['nuclei']:
                a = r1[nuc]
                b = r2[nuc]
                if not (np.isnan(a) or np.isnan(b)):
                    g['v1'].append(a)
                    g['v2'].append(b)

    offsets = {}
    print(f"{'Isotope':<6} {'Points':>8} {'Offset (ppm)':>14} {'RMSD before':>14} {'RMSD after':>14}")
    print("-" * 62)
    for iso, g in groups.items():
        if not g['v1']:
            offsets[iso] = 0.0
            print(f"{iso:<6} {'-':>8}   no data → offset 0.0000")
            continue
        a = np.array(g['v1'])
        b = np.array(g['v2'])
        diff = b - a
        offset = diff.mean()
        rmsd_before = np.sqrt(np.mean(diff**2))
        rmsd_after  = np.sqrt(np.mean((diff - offset)**2))
        print(f"{iso:<6} {len(a):8} {offset:+14.4f} {rmsd_before:14.4f} {rmsd_after:14.4f}")
        offsets[iso] = offset
    print()
    return offsets


# ------------------------------------------------------------------
# 3. Apply offsets (in-place on df1)
# ------------------------------------------------------------------
def apply_isotope_offsets(df, offsets):
    mapping = {
        '1H':  ['HA', 'HN'],
        '13C': ['CA', 'CB', 'C'],
        '15N': ['N']
    }
    df = df.copy()
    for iso, nuclei in mapping.items():
        off = offsets.get(iso, 0.0)
        for nuc in nuclei:
            if nuc in df.columns:
                mask = df[nuc].notna()
                df.loc[mask, nuc] += off
    return df


# ------------------------------------------------------------------
# 4. Write output with *exactly* the same formatting as input
# ------------------------------------------------------------------
def write_identical_format(df_corr, raw_lines_input, offsets, output_path):
    # Mapping from column name → position in the original line
    col_order = ['HA', 'CA', 'HN', 'N', 'C', 'CB']

    # Build header with applied offsets
    offset_str = " | ".join([f"{iso} {off:+.4f}" for iso, off in offsets.items()])
    with open(output_path, 'w') as out:
        out.write(f"#RESID RESNAME HA CA HN N C CB   (isotope offsets: {offset_str})\n")

        for original_line in raw_lines_input:
            fields = original_line.split()
            resid = int(fields[0])
            resname = fields[1]

            # Re-build the six value fields with corrected numbers (or original "nan")
            new_values = []
            for col in col_order:
                val = df_corr.loc[resid, col]
                if np.isnan(val):
                    new_values.append("nan")
                else:
                    new_values.append(f"{val:.3f}")

            # Reconstruct line with exactly the same spacing as original
            # Original format: 4-digit resid, 6-char resname, then 6×8-char fields
            line = (f"{resid:4d} {resname:>6} " +
                    " ".join(f"{v:>8}" if v != "nan" else "     nan" for v in new_values))
            out.write(line + "\n")

    print(f"Corrected file written with identical formatting → {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(file_to_correct, reference_file, output_file=None):
    print(f"Reading file to correct : {file_to_correct}")
    df1, lines1 = read_chemical_shift_file(file_to_correct)
    print(f"Reading reference file   : {reference_file}")
    df2, _      = read_chemical_shift_file(reference_file)

    offsets = compute_isotope_offsets(df1, df2)

    df_corrected = apply_isotope_offsets(df1, offsets)

    if output_file is None:
        stem = Path(file_to_correct).stem
        output_file = f"{stem}_isotope_corrected.dat"

    write_identical_format(df_corrected, lines1, offsets, output_file)


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nUsage:")
        print("  python correct_shifts_isotope_exact.py <file_to_correct.dat> <reference.dat> [output.dat]")
        print("  Output has exactly the same column widths and spacing as the input.\n")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
