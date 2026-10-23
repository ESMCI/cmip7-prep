"""What validate_cmor_output.py's exit codes mean.

Defined here rather than in the script so that the script and
run_reference_case.py, which runs it, cannot disagree about them.

The distinction that matters is between validation that ran and found problems
and validation that could not run.  Both used to exit 1 -- Python exits 1 on an
uncaught exception, and 1 was also "found problems" -- so a crash looked like a
report, and a resume skipped it as though it had been done.
"""

RAN_AND_PASSED = 0
RAN_AND_FOUND_PROBLEMS = 1
COULD_NOT_RUN = 3
