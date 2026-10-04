"""Tests for planning a full-chain reference-case run."""

import pytest

from cmip7_prep.reference_run import (
    REALM_COMPONENT,
    STAGES,
    build_plan,
    frequencies_for,
    history_dir,
    realms_for,
)


@pytest.fixture(name="case_dir")
def case_dir_fixture(tmp_path):
    """Return a case directory with every component's history directory."""
    for component in sorted(set(REALM_COMPONENT.values())):
        (tmp_path / component / "hist").mkdir(parents=True)
    return tmp_path


def _plan(case_dir, outdir, **kwargs):
    """Call build_plan with a model, which it requires.

    The model has no default on purpose: the wrong one yields plausible but
    wrong output, so a caller must say which it means.
    """
    kwargs.setdefault("model", "noresm")
    kwargs.setdefault("atmos_res", "ne16")
    kwargs.setdefault("experiment", "piControl")
    return build_plan(case_dir, outdir, **kwargs)


def _command_of(plan, key):
    """Return the command of the step with one key."""
    for step in plan.steps:
        if step.key == key:
            return step.command
    raise AssertionError(f"no step {key!r} in {[s.key for s in plan.steps]}")


# ------------------------------------------------------------------- lookups


def test_realms_come_from_the_tables():
    """The realms planned for are the ones the include-pattern table declares."""
    assert realms_for("noresm") == [
        "atmos",
        "atmosChem",
        "aerosol",
        "land",
        "seaIce",
        "landIce",
    ]


def test_every_realm_has_a_component_directory():
    """No declared realm is missing from the component map.

    A realm present in the tables but absent here would be silently skipped,
    so the two must stay in step.
    """
    for model in ("noresm", "cesm"):
        for realm in realms_for(model):
            assert realm in REALM_COMPONENT, f"{model}/{realm} has no component"


def test_history_dir_follows_the_archive_layout(tmp_path):
    """History lives under <case>/<component>/hist."""
    assert history_dir(tmp_path, "seaIce") == tmp_path / "ice" / "hist"
    assert history_dir(tmp_path, "atmosChem") == tmp_path / "atm" / "hist"


# ---------------------------------------------------------------- plan shape


class TestPlanShape:
    """Tests for which steps a plan contains."""

    def test_covers_every_realm_by_default(self, case_dir, tmp_path):
        """One timeseries step per realm, when no realms are named."""
        plan = _plan(case_dir, tmp_path / "out")
        timeseries = plan.for_stage("timeseries")
        assert {step.realm for step in timeseries} == set(realms_for("noresm"))
        assert len(timeseries) == 6

    def test_cmor_and_validate_are_per_frequency(self, case_dir, tmp_path):
        """CMOR and validation run once per realm and frequency."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"])
        assert [step.key for step in plan.for_stage("cmor")] == [
            "cmor-seaIce-day",
            "cmor-seaIce-mon",
        ]
        assert [step.key for step in plan.for_stage("validate")] == [
            "validate-seaIce-day",
            "validate-seaIce-mon",
        ]

    def test_realms_can_be_narrowed(self, case_dir, tmp_path):
        """Naming realms restricts the plan to them."""
        plan = _plan(case_dir, tmp_path / "out", realms=["land", "seaIce"])
        assert {step.realm for step in plan.steps} == {"land", "seaIce"}

    def test_frequencies_can_be_narrowed(self, case_dir, tmp_path):
        """Naming frequencies restricts each realm to the ones it declares."""
        plan = _plan(
            case_dir, tmp_path / "out", realms=["atmos"], frequencies=["mon", "day"]
        )
        assert [step.frequency for step in plan.for_stage("cmor")] == ["day", "mon"]

    def test_frequency_a_realm_does_not_declare_is_skipped(self, case_dir, tmp_path):
        """Asking seaIce for 1hr skips it rather than planning a doomed step."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"], frequencies=["1hr"])
        assert not plan.steps
        assert any("seaIce" in reason for reason in plan.skipped)

    def test_stages_can_be_narrowed(self, case_dir, tmp_path):
        """Only the requested stages are planned."""
        plan = _plan(
            case_dir, tmp_path / "out", realms=["seaIce"], stages=["timeseries"]
        )
        assert {step.stage for step in plan.steps} == {"timeseries"}

    def test_unknown_stage_is_rejected(self, case_dir, tmp_path):
        """A misspelled stage fails immediately, not mid-run."""
        with pytest.raises(ValueError, match="Unknown stage"):
            _plan(case_dir, tmp_path / "out", stages=["timeseries", "cmorise"])

    def test_missing_component_is_skipped_not_fatal(self, tmp_path):
        """A case without every component plans what it can and says so.

        An archived case need not hold every component, so a missing directory
        is reported rather than ending the run.
        """
        case = tmp_path / "case"
        (case / "ice" / "hist").mkdir(parents=True)
        plan = _plan(case, tmp_path / "out")
        assert {step.realm for step in plan.steps} == {"seaIce"}
        assert len(plan.skipped) == 5
        assert all("no history directory" in reason for reason in plan.skipped)

    def test_all_stages_present_by_default(self, case_dir, tmp_path):
        """The default plan runs all three stages."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"])
        assert {step.stage for step in plan.steps} == set(STAGES)


# -------------------------------------------------------------- command form


class TestCommands:
    """Tests for the commands a plan builds."""

    def test_timeseries_command_points_at_the_right_history_dir(
        self, case_dir, tmp_path
    ):
        """Each realm reads from its own component's history directory."""
        plan = _plan(case_dir, tmp_path / "out", realms=["land"])
        command = _command_of(plan, "timeseries-land")
        assert "--inputdir" in command
        assert command[command.index("--inputdir") + 1] == str(
            case_dir / "lnd" / "hist"
        )

    def test_timeseries_passes_every_declared_frequency(self, case_dir, tmp_path):
        """One timeseries run covers all of a realm's frequencies."""
        plan = _plan(case_dir, tmp_path / "out", realms=["land"])
        command = _command_of(plan, "timeseries-land")
        at = command.index("--frequency")
        assert command[at + 1 : at + 5] == frequencies_for("noresm", "land")

    def test_years_are_passed_through(self, case_dir, tmp_path):
        """--years reaches gen_timeseries.py as --years-spec."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"], years="1526:1527:2")
        command = _command_of(plan, "timeseries-seaIce")
        assert command[command.index("--years-spec") + 1] == "1526:1527:2"

    def test_no_years_spec_when_not_asked_for(self, case_dir, tmp_path):
        """Without --years the flag is absent, so the script's default applies."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"])
        assert "--years-spec" not in _command_of(plan, "timeseries-seaIce")

    def test_ice_sheet_only_for_landice(self, case_dir, tmp_path):
        """landIce gets --ice-sheet; other realms do not."""
        plan = _plan(case_dir, tmp_path / "out", realms=["landIce", "seaIce"])
        assert "--ice-sheet" in _command_of(plan, "timeseries-landIce")
        assert "--ice-sheet" not in _command_of(plan, "timeseries-seaIce")

    def test_ice_sheet_can_be_chosen(self, case_dir, tmp_path):
        """The ice sheet given is the one passed on."""
        plan = _plan(case_dir, tmp_path / "out", realms=["landIce"], ice_sheet="ais")
        command = _command_of(plan, "timeseries-landIce")
        assert command[command.index("--ice-sheet") + 1] == "ais"

    def test_cmor_reads_the_timeseries_the_first_stage_writes(self, case_dir, tmp_path):
        """The CMOR step's --tsdir is the timeseries step's --outputdir."""
        out = tmp_path / "out"
        plan = _plan(case_dir, out, realms=["seaIce"])
        wrote = _command_of(plan, "timeseries-seaIce")
        reads = _command_of(plan, "cmor-seaIce-mon")
        assert (
            wrote[wrote.index("--outputdir") + 1] == reads[reads.index("--tsdir") + 1]
        )

    def test_validate_reads_where_cmor_writes(self, case_dir, tmp_path):
        """Validation's --root-output-path is the CMOR step's --outdir."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"])
        wrote = _command_of(plan, "cmor-seaIce-mon")
        reads = _command_of(plan, "validate-seaIce-mon")
        assert (
            wrote[wrote.index("--outdir") + 1]
            == reads[reads.index("--root-output-path") + 1]
        )

    def test_validate_is_strict(self, case_dir, tmp_path):
        """Validation fails the run on missing variables or log errors."""
        plan = _plan(case_dir, tmp_path / "out", realms=["seaIce"])
        assert "--strict" in _command_of(plan, "validate-seaIce-mon")

    def test_resolution_and_experiment_reach_cmor(self, case_dir, tmp_path):
        """The atmosphere resolution and experiment are passed to the CMOR step.

        The CMOR step is given the atmosphere resolution rather than the grid
        it will regrid from: cmor_driver.py derives that itself from --model
        and --realm, so the derivation has one home. See test_grids.py.
        """
        plan = _plan(
            case_dir,
            tmp_path / "out",
            realms=["atmos"],
            atmos_res="ne30",
            experiment="historical",
        )
        command = _command_of(plan, "cmor-atmos-mon")
        assert command[command.index("--atmos-res") + 1] == "ne30"
        assert command[command.index("--experiment") + 1] == "historical"

    def test_every_step_has_a_distinct_log_name(self, case_dir, tmp_path):
        """Logs cannot overwrite one another."""
        plan = _plan(case_dir, tmp_path / "out")
        names = [step.log_name for step in plan.steps]
        assert len(names) == len(set(names))


@pytest.mark.parametrize("omitted", ["model", "experiment"])
def test_case_properties_are_required(omitted):
    """build_plan refuses to guess a property of the case.

    Model and experiment change what the output means while leaving it looking
    valid, so neither may be defaulted.  The atmosphere resolution is not here
    because it is only needed by some realms; see
    TestAtmosResolutionIsConditional.
    """
    supplied = {
        "model": "noresm",
        "atmos_res": "ne16",
        "experiment": "piControl",
    }
    del supplied[omitted]
    with pytest.raises(TypeError, match=omitted):
        build_plan("case", "out", **supplied)


class TestAtmosResolutionIsConditional:
    """The atmosphere resolution is asked for only when a realm uses it."""

    def test_not_needed_for_realms_with_their_own_grid(self, case_dir, tmp_path):
        """Sea ice and land ice plan fully without one."""
        plan = build_plan(
            case_dir,
            tmp_path / "out",
            model="noresm",
            experiment="piControl",
            realms=["seaIce", "landIce"],
        )
        assert plan.steps
        for step in plan.for_stage("cmor"):
            assert "--atmos-res" not in step.command

    @pytest.mark.parametrize("realm", ["atmos", "land"])
    def test_needed_for_realms_on_the_atmosphere_grid(self, case_dir, tmp_path, realm):
        """A realm on that grid refuses to plan without one, and names itself."""
        with pytest.raises(ValueError, match=f"needed for \\['{realm}'\\]"):
            build_plan(
                case_dir,
                tmp_path / "out",
                model="noresm",
                experiment="piControl",
                realms=[realm],
            )

    def test_the_whole_matrix_names_every_realm_that_needs_it(self, case_dir, tmp_path):
        """Planning every realm without one lists the four that require it."""
        with pytest.raises(ValueError, match="atmos.*atmosChem.*aerosol.*land"):
            build_plan(
                case_dir, tmp_path / "out", model="noresm", experiment="piControl"
            )

    def test_only_atmosphere_realms_carry_the_flag(self, case_dir, tmp_path):
        """With one supplied, only the realms that use it are given it."""
        plan = _plan(case_dir, tmp_path / "out", realms=["atmos", "seaIce"])
        atm = _command_of(plan, "cmor-atmos-mon")
        sea = _command_of(plan, "cmor-seaIce-mon")
        assert atm[atm.index("--atmos-res") + 1] == "ne16"
        assert "--atmos-res" not in sea


# ------------------------------------------- forwarding the grid decision


class TestGridIsNotDecidedHere:
    """The plan never decides which grid a run regrids from."""

    def test_no_step_carries_a_derived_grid(self, case_dir, tmp_path):
        """No command names an input grid such as tnx1v4.

        cmor_driver.py derives that from --model and --realm, so a plan that
        passed one would mean two places deciding the same thing.
        """
        plan = _plan(case_dir, tmp_path / "out")
        for step in plan.steps:
            assert "--resolution" not in step.command
            assert "tnx1v4" not in step.command
            assert "tx2_3v2" not in step.command

    def test_landice_runs_every_stage_without_extra_input(self, case_dir, tmp_path):
        """land ice needs no grid input of its own."""
        plan = _plan(case_dir, tmp_path / "out", realms=["landIce"])
        assert {step.stage for step in plan.steps} == set(STAGES)
        assert not plan.skipped
        command = _command_of(plan, "cmor-landIce-yr")
        assert "--atmos-res" not in command
        assert command[command.index("--ice-sheet") + 1] == "gris"
