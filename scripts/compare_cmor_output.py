#!/usr/bin/env python
"""Compare two CMOR output trees, file by file and value by value.

For checking that a parallel run produced the same output as a serial one::

    python compare_cmor_output.py <dir-a> <dir-b>

Matches files by their path below the CMIP7 root, so the two trees may sit
anywhere.  Reports files present in only one tree, and for the rest whether
every variable holds the same values.

Two global attributes are expected to differ between any two runs and are not
compared: creation_date and the history line CMOR appends.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr


def relative_paths(root: Path) -> dict[str, Path]:
    """Return every .nc file under root, keyed by its path below CMIP7."""
    found = {}
    for path in root.rglob("*.nc"):
        parts = path.parts
        if "CMIP7" in parts:
            key = str(Path(*parts[parts.index("CMIP7") :]))
        else:
            key = path.name
        found[key] = path
    return found


def same_values(path_a: Path, path_b: Path) -> list[str]:
    """Return the variables whose values differ, or an empty list."""
    differing = []
    with xr.open_dataset(path_a) as a, xr.open_dataset(path_b) as b:
        for name in a.data_vars:
            if name not in b:
                differing.append(f"{name} (missing in the second file)")
                continue
            left = np.asarray(a[name])
            right = np.asarray(b[name])
            if left.shape != right.shape:
                differing.append(f"{name} ({left.shape} against {right.shape})")
            elif np.issubdtype(left.dtype, np.number):
                # allclose rather than equality: the same arithmetic in a
                # different process can differ in the last bit.
                if not np.allclose(left, right, equal_nan=True):
                    differing.append(name)
            elif not np.array_equal(left, right):
                # Times and anything else not numeric -- allclose cannot
                # subtract a cftime date, so compare them exactly.
                differing.append(name)
    return differing


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root_a, root_b = (Path(p).expanduser().resolve() for p in sys.argv[1:3])
    files_a = relative_paths(root_a)
    files_b = relative_paths(root_b)

    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    shared = sorted(set(files_a) & set(files_b))

    # Both trees are usually called "cmor", so name them by something that
    # differs.  Falling back to the full path when even the parents match.
    label_a, label_b = root_a.parent.name, root_b.parent.name
    if label_a == label_b:
        label_a, label_b = str(root_a), str(root_b)

    print(f"A = {root_a}\nB = {root_b}\n")
    print(f"{len(files_a)} file(s) under A, {len(files_b)} under B\n")

    for key in only_a:
        print(f"ONLY IN A ({label_a}): {key}")
    for key in only_b:
        print(f"ONLY IN B ({label_b}): {key}")

    differed = 0
    for key in shared:
        problems = same_values(files_a[key], files_b[key])
        if problems:
            differed += 1
            print(f"DIFFERS: {key}")
            for problem in problems:
                print(f"    {problem}")

    print(
        f"\n{len(shared)} file(s) in both, {differed} with differing values; "
        f"{len(only_a)} only in A ({label_a}), {len(only_b)} only in B ({label_b})"
    )
    return 1 if (differed or only_a or only_b) else 0


if __name__ == "__main__":
    sys.exit(main())
