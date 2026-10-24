"""What the exit codes of the scripts run_reference_case.py runs mean.

Defined here rather than in each script so that the script and the caller
cannot disagree about them.

validate_cmor_output.py and gen_timeseries.py both have an outcome that is
neither success nor failure, and the caller treats those differently: it
records them and carries on, rather than stopping the run.
"""

RAN_AND_PASSED = 0

# validate_cmor_output.py.  Validation that found problems has to be
# distinguishable from validation that could not run, since Python exits 1 on
# an uncaught exception.
RAN_AND_FOUND_PROBLEMS = 1
COULD_NOT_RUN = 3

# gen_timeseries.py.  A frequency with no time series at all, which no later
# step can work from.  Rerunning cannot produce output the case never wrote, so
# the run notes it and goes on to the next realm.
A_FREQUENCY_IS_EMPTY = 4
