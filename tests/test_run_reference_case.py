"""Tests for scripts/run_reference_case.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from run_reference_case import (
    completed_path,
    counts_as_done,
    outcome_label,
    period_tag,
    read_completed,
    record_completed,
    should_stop,
    step_log_dir,
)


class TestCountsAsDone:
    """Tests for which finished steps a later resume should skip."""

    @pytest.mark.parametrize("stage", ["timeseries", "cmor", "validate"])
    def test_a_step_that_succeeded_is_done(self, stage):
        """Whatever the stage."""
        assert counts_as_done(stage, 0) is True

    @pytest.mark.parametrize("stage", ["timeseries", "cmor"])
    def test_a_step_that_failed_is_not_done(self, stage):
        """It has to be run again; nothing downstream can proceed without it."""
        assert counts_as_done(stage, 1) is False

    def test_validation_that_found_problems_is_done(self):
        """Exit 1 means it ran and reported. --strict fails it while any
        requested variable is missing, so recording it only on success meant
        every resume repeated all of it."""
        assert counts_as_done("validate", 1) is True

    def test_validation_that_could_not_run_is_not_done(self):
        """Exit 3 is the validator saying it crashed -- nothing to show, so it
        has to be tried again rather than recorded."""
        assert counts_as_done("validate", 3) is False

    @pytest.mark.parametrize("returncode", [2, 137, -15])
    def test_any_other_ending_is_not_done(self, returncode):
        """An argument error, a kill, a signal: none of them ran."""
        assert counts_as_done("validate", returncode) is False

    def test_a_time_series_step_missing_a_frequency_is_done(self):
        """Exit 4 says a frequency has no output, which a rerun cannot change."""
        assert counts_as_done("timeseries", 4) is True

    def test_exit_4_from_another_stage_is_not_done(self):
        """Only the time series step gives exit 4 that meaning."""
        assert counts_as_done("cmor", 4) is False


class TestOutcomeLabel:
    """Tests for how a finished step reads in the summary table."""

    @pytest.mark.parametrize("stage", ["timeseries", "cmor", "validate"])
    def test_success_is_ok(self, stage):
        """Whatever the stage."""
        assert outcome_label(stage, 0) == "ok"

    def test_a_time_series_step_missing_a_frequency_is_a_warning(self):
        """Not ok, because something is absent; not a failure, because the
        case never wrote it."""
        assert outcome_label("timeseries", 4) == "warning"

    def test_validation_finding_problems_is_a_report(self):
        """Reporting is what it is for."""
        assert outcome_label("validate", 1) == "reported problems"

    @pytest.mark.parametrize(
        "stage,returncode", [("timeseries", 1), ("cmor", 4), ("validate", 3)]
    )
    def test_anything_else_is_a_failure(self, stage, returncode):
        """Including exit 4 from a stage that does not define it."""
        assert outcome_label(stage, returncode) == f"FAILED ({returncode})"


class TestShouldStop:
    """Tests for which results end the rest of the run."""

    @pytest.mark.parametrize("stage", ["timeseries", "cmor", "validate"])
    def test_success_never_stops(self, stage):
        """Whatever the stage."""
        assert should_stop(stage, 0, keep_going=False) is False

    def test_a_time_series_step_missing_a_frequency_does_not_stop(self):
        """The remaining realms are unaffected by it."""
        assert should_stop("timeseries", 4, keep_going=False) is False

    def test_a_failed_time_series_step_stops(self):
        """Everything after it would read input that is not there."""
        assert should_stop("timeseries", 1, keep_going=False) is True

    def test_keep_going_overrides_a_failure(self):
        """That is what it is for."""
        assert should_stop("timeseries", 1, keep_going=True) is False


class TestPeriodTag:
    """Tests for naming the period a run covers."""

    def test_a_tag_from_the_years(self):
        """First and last, whatever the increment between them."""
        assert period_tag("1:50:10") == "1-50"
        assert period_tag("51:100:10") == "51-100"
        assert period_tag("1526:1535:5") == "1526-1535"

    def test_no_years_means_no_tag(self):
        """Behaviour then is exactly as it was before periods existed."""
        assert period_tag(None) is None
        assert period_tag("") is None

    def test_two_periods_get_different_tags(self):
        """Which is the whole point: they must not share a progress file."""
        assert period_tag("1:50:10") != period_tag("51:100:10")


class TestWhereProgressAndLogsGo:
    """Tests that two periods keep their own records."""

    def test_progress_is_per_period(self, tmp_path):
        """Years 51-100 must not read years 1-50 as already done."""
        first = completed_path(tmp_path, "1-50")
        second = completed_path(tmp_path, "51-100")
        assert first != second
        record_completed(first, {"cmor-atmos-mon"})
        assert read_completed(first) == {"cmor-atmos-mon"}
        assert read_completed(second) == set()

    def test_step_logs_are_per_period(self, tmp_path):
        """So finishing one period cannot overwrite another's logs."""
        assert step_log_dir(tmp_path, "1-50") != step_log_dir(tmp_path, "51-100")
        assert step_log_dir(tmp_path, "1-50") == tmp_path / "logs" / "1-50"

    def test_without_a_period_nothing_changes(self, tmp_path):
        """The old paths, for a run given no --years."""
        assert completed_path(tmp_path, None) == tmp_path / "completed_steps.json"
        assert step_log_dir(tmp_path, None) == tmp_path / "logs"

    def test_a_missing_progress_file_is_not_an_error(self, tmp_path):
        """A first run has nothing recorded yet."""
        assert read_completed(completed_path(tmp_path, "1-50")) == set()
