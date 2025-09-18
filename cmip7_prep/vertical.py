# cmip7_prep/vertical.py
def to_plev19(ds, var, vdef):
    # Use model hybrid coefficients + surface pressure to compute pressure
    # then interpolate log-pressure to requested plev19 (Pa).
    # Return dataset with dim replaced by 'plev'.
    return ds

