"""tests/test_cmor_writer_helpers.py"""

import numpy as np
import pytest

from cmip7_prep.cmor_utils import encode_time_to_num

cftime = pytest.importorskip("cftime")


def test_encode_time_to_num_with_cftime_no_leap():
    """test  0001-01-16 12:00, 0001-02-15 12:00 in noleap"""
    t0 = cftime.DatetimeNoLeap(1, 1, 16, 12, 0, 0)
    t1 = cftime.DatetimeNoLeap(1, 2, 15, 12, 0, 0)
    arr = np.array([t0, t1], dtype=object)

    out = encode_time_to_num(arr, units="days since 0001-01-01", calendar="noleap")

    assert out.shape == (2,)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert np.all(np.diff(out) > 0)  # strictly increasing
