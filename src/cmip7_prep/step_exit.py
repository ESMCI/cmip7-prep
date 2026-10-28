"""Define the exit codes run_reference_case.py reads from each script.

They live here rather than in each script so that a script and its caller
cannot disagree about them.

Some scripts can end in a way that is neither success nor failure. The caller
records those and carries on rather than stopping the run.
"""

RAN_AND_PASSED = 0

# validate_cmor_output.py.  Validation that found problems has to be
# distinguishable from validation that could not run, since Python exits 1 on
# an uncaught exception.
RAN_AND_FOUND_PROBLEMS = 1
COULD_NOT_RUN = 3

# gen_timeseries.py. A frequency came out with no time series at all, which no
# later step can work from. The case never wrote the history files it needed.
A_FREQUENCY_IS_EMPTY = 4

# cmor_driver.py. The mapping file has none of the variables the data request
# asks for, so there was nothing to write. Exit 1 means every variable that was
# tried failed.
NOTHING_TO_WRITE = 5
