#!/usr/bin/env python3
"""
convert_mr_final.py

Converts NMR assign restraints (.mr) to simplified format.

Rules:
- Keep only assign blocks with at most one "or".
- Replace '#' -> '*'.
- If an "or" lists atom names that differ only by trailing digits (HA1, HA2) -> collapse into wildcard (HA*).
- If the "or" lists are identical (same resid and name) -> treat as single atom (no wildcard added).
- Otherwise, expand alternatives into multiple output lines.
- Output format: <resid_left> <name_left> <resid_right> <name_right> <d1> <d2> <d3>
"""

import re
from pathlib import Path

INPUT = "input.mr"
OUTPUT = "output.txt"


def strip_trailing_digits_base(name):
    """Return base (without trailing digits) and whether digits were present."""
    m = re.match(r'^([A-Za-z*]+?)(\d+)$', name)
    if m:
        return m.group(1), True
    return name, False


def compress_names_if_numeric_variants(names):
    """
    If all names (after replacing '#') share the same base (differs only by trailing digits),
    return single wildcard form (base+'*'). Else return None.
    """
    names = [n.replace('#', '*') for n in names]
    # identical names? don't collapse
    if len(set(names)) == 1:
        return names[0]
    bases = [strip_trailing_digits_base(n)[0] for n in names]
    if len(set(bases)) == 1:
        return bases[0] + '*'
    return None


def parse_assign_blocks(content):
    """Extract assign ... ! blocks (handles multiline)."""
    pattern = re.compile(r'assign\s+(.+?)!', re.DOTALL | re.IGNORECASE)
    return [m.group(0) for m in pattern.finditer(content)]


def process_block(block):
    """Process a single assign block and return list of formatted restraint lines."""
    # skip if more than one ' or '
    if block.lower().count(' or ') > 1:
        return []

    # numeric triplet (d1, d2, d3)
    before_excl = block.split('!')[0]
    triplet_match = re.search(r'([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)', before_excl)
    if not triplet_match:
        return []
    d1, d2, d3 = triplet_match.group(1), triplet_match.group(2), triplet_match.group(3)

    # extract atoms
    atom_re = re.compile(r'\(resid\s+(\d+)\s+and\s+name\s+([A-Za-z0-9#*]+)\)', re.IGNORECASE)
    matches = list(atom_re.finditer(before_excl))
    if len(matches) < 2:
        return []

    info = [(m.start(), m.end(), m.group(1), m.group(2)) for m in matches]
    or_index = before_excl.lower().find(' or ')
    if or_index == -1:
        # simple pair
        left_resid, left_name = info[0][2], info[0][3].replace('#', '*')
        right_resid, right_name = info[1][2], info[1][3].replace('#', '*')
        return [f"{left_resid} {left_name} {right_resid} {right_name} {d1} {d2} {d3}"]

    # find which pair 'or' sits between
    or_between = None
    for i in range(len(info) - 1):
        if info[i][1] <= or_index <= info[i + 1][0]:
            or_between = i
            break
    if or_between is None:
        return []

    out = []
    if or_between == 0 and len(info) >= 3:
        # left-side alternatives
        left_choices = [info[0], info[1]]
        right = info[2]
        # check if identical alternatives
        if left_choices[0][2:] == left_choices[1][2:]:
            # same resid and name -> just use one
            left_resid, left_name = left_choices[0][2], left_choices[0][3].replace('#', '*')
            right_resid, right_name = right[2], right[3].replace('#', '*')
            return [f"{left_resid} {left_name} {right_resid} {right_name} {d1} {d2} {d3}"]

        left_names = [a[3].replace('#', '*') for a in left_choices]
        left_resid = left_choices[0][2]
        right_resid, right_name = right[2], right[3].replace('#', '*')

        compressed = compress_names_if_numeric_variants(left_names)
        if compressed:
            out.append(f"{left_resid} {compressed} {right_resid} {right_name} {d1} {d2} {d3}")
        else:
            for n in left_names:
                out.append(f"{left_resid} {n} {right_resid} {right_name} {d1} {d2} {d3}")
        return out

    elif or_between == len(info) - 2 and len(info) >= 3:
        # right-side alternatives
        left = info[0]
        right_choices = [info[-2], info[-1]]
        # identical alternatives?
        if right_choices[0][2:] == right_choices[1][2:]:
            left_resid, left_name = left[2], left[3].replace('#', '*')
            right_resid, right_name = right_choices[0][2], right_choices[0][3].replace('#', '*')
            return [f"{left_resid} {left_name} {right_resid} {right_name} {d1} {d2} {d3}"]

        right_names = [a[3].replace('#', '*') for a in right_choices]
        right_resid = right_choices[0][2]
        left_resid, left_name = left[2], left[3].replace('#', '*')

        compressed = compress_names_if_numeric_variants(right_names)
        if compressed:
            out.append(f"{left_resid} {left_name} {right_resid} {compressed} {d1} {d2} {d3}")
        else:
            for n in right_names:
                out.append(f"{left_resid} {left_name} {right_resid} {n} {d1} {d2} {d3}")
        return out

    return []


def convert_file(input_path=INPUT, output_path=OUTPUT):
    p = Path(input_path)
    if not p.exists():
        print(f"Input file '{input_path}' not found.")
        return

    content = p.read_text()
    blocks = parse_assign_blocks(content)
    out_lines = []
    for b in blocks:
        lines = process_block(b)
        if lines:
            out_lines.extend(lines)

    # deduplicate preserving order
    seen = set()
    uniq = []
    for L in out_lines:
        if L not in seen:
            seen.add(L)
            uniq.append(L)

    Path(output_path).write_text("\n".join(uniq) + ("\n" if uniq else ""))
    print(f"✅ Wrote {len(uniq)} restraints to {output_path}")


if __name__ == "__main__":
    convert_file()

