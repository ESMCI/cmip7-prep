#!/usr/bin/env python
"""Can ESMF be used inside worker processes at all?

cmor_driver.py --workers N regrids in several processes at once, and every
regrid fails with

    ESMCI::VM::getCurrent() Internal error: Could not determine current VM

This asks the narrow question behind that, without dask, CMOR or any of our
own code in the way: start plain worker processes, and have each one build a
small xesmf regridder.

    python esmf_worker_check.py          # 2 workers, spawned
    python esmf_worker_check.py 4 fork   # 4 workers, forked

If the workers succeed, ESMF is fine in worker processes and the fault is in
how cmor_driver.py sets them up.  If they fail the same way, ESMF cannot be
used like this here, and variables have to be parallelised by running separate
cmor_driver.py processes rather than workers inside one.
"""

import multiprocessing as mp
import sys


def build_regridder(tag):
    """Build a small regridder, reporting what happened rather than raising."""
    try:
        import numpy as np
        import xarray as xr
        import xesmf as xe

        src = xr.Dataset(
            {
                "lat": ("lat", np.linspace(-89.0, 89.0, 10)),
                "lon": ("lon", np.linspace(0.0, 350.0, 12)),
            }
        )
        dst = xr.Dataset(
            {
                "lat": ("lat", np.linspace(-89.0, 89.0, 5)),
                "lon": ("lon", np.linspace(0.0, 350.0, 6)),
            }
        )
        regridder = xe.Regridder(src, dst, "bilinear")
        return f"{tag}: OK, weights {regridder.weights.data.shape}"
    except Exception as exc:  # pylint: disable=broad-except
        return f"{tag}: FAILED {type(exc).__name__}: {exc}"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    method = sys.argv[2] if len(sys.argv) > 2 else "spawn"

    # In the parent first, so a failure there is not mistaken for one in a
    # worker, and so we know whether importing xesmf here is what breaks them.
    print("parent:", build_regridder("parent"), flush=True)

    context = mp.get_context(method)
    with context.Pool(workers) as pool:
        for line in pool.map(build_regridder, [f"worker{i}" for i in range(workers)]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
