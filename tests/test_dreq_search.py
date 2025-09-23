"""Unit tests for cmip7_prep.dreq_search."""

from pathlib import Path

from cmip7_prep.dreq_search import find_variables_by_prefix


def _write_min_csv(p: Path) -> Path:
    """Write a small CSV tailored to your header schema and return its path."""
    text = (
        "CMIP6 Compound Name,CMIP7 Compound Name,Branded Variable Name,"
        "Physical Parameter,CMIP6 Table (legacy),CMIP7 Variable Groups,"
        "CMIP7 Frequency\n"
        # Amon, monthly, baseline
        "Amon.tas,,tas_tavg-u-hxy-u,tas,Amon,baseline_monthly,monthly\n"
        "Amon.ta,,ta_tavg-u-h3d,ta,Amon,baseline_monthly,monthly\n"
        # Amon, daily (should be excluded by monthly filters)
        "Amon.tauu,,tauu,tauu,Amon,baseline_daily,daily\n"
        # AERmon, monthly
        "AERmon.mmrbc,,mmrbc_something,mmrbc,AERmon,baseline_monthly,monthly\n"
        # Amon, monthly, another baseline var
        "Amon.zg,,zg,zg,Amon,baseline_monthly,monthly\n"
        # Amon, monthly, physical missing → fallback to Branded
        "Amon.foo,,name_from_branded,,Amon,baseline_monthly,monthly\n"
        # Amon, table missing but compound present (dot)
        "Amon.pr,,pr,pr,,baseline_monthly,monthly\n"
        # Amon, table missing but compound present (space)
        "Amon tas,,tas_alt,tas_alt,,baseline_monthly,monthly\n"
        # Amon, aod example where Branded is long, Physical is short
        "Amon.aod550volso4,,aod550volso4_tavg-u-hxy-u,aod550volso4,Amon,baseline_monthly,monthly\n"
    )
    f = p / "dreq_min.csv"
    f.write_text(text, encoding="utf-8")
    return f


def test_dotted_table_all_returns_physical_parameters(tmp_path: Path):
    """'Amon.' returns all Amon variables as short names (Physical Parameter)."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(csv_path, "Amon.")
    # Expect: tas, ta, tauu, zg, name_from_branded (fallback), pr, tas_alt, aod550volso4
    assert got == sorted(
        [
            "tas",
            "ta",
            "tauu",
            "zg",
            "name_from_branded",
            "pr",
            "tas_alt",
            "aod550volso4",
        ]
    )


def test_prefix_filter_and_group_subset(tmp_path: Path):
    """'Amon.ta' with include_groups filters out daily tauu, keeping monthly ta/tas."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(
        csv_path,
        "Amon.ta",
        include_groups={"baseline_monthly"},
    )
    # 'ta' and 'tas' match the 'ta' prefix and are monthly; 'tauu' is baseline_daily → excluded.
    assert got == ["ta", "tas", "tas_alt"]


def test_exact_match_in_other_table(tmp_path: Path):
    """Exact match should return only the exact physical parameter from that table."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(csv_path, "AERmon.mmrbc", exact=True)
    assert got == ["mmrbc"]


def test_where_filter_on_frequency(tmp_path: Path):
    """where={'cmip7 frequency': 'monthly'} keeps only monthly rows."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(
        csv_path,
        "Amon.",
        where={"cmip7 frequency": "monthly"},
    )
    # daily 'tauu' removed
    assert "tauu" not in got
    # sanity: others remain
    for v in ["tas", "ta", "zg", "name_from_branded", "pr", "tas_alt"]:
        assert v in got


def test_match_on_branded_return_physical(tmp_path: Path):
    """Match using Branded Variable Name but return Physical Parameter (short)."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(
        csv_path,
        "Amon.aod550",  # matches Branded 'aod550volso4_tavg-u-hxy-u'
        match_field="Branded Variable Name",
        return_field="Physical Parameter",
    )
    assert got == ["aod550volso4"]


def test_table_fallback_from_compound_when_table_missing(tmp_path: Path):
    """Rows without 'CMIP6 Table (legacy)' fall back to the compound for table."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(csv_path, "Amon.")
    # 'pr' row has empty table but compound 'Amon.pr'; 'tas_alt' has 'Amon tas'
    assert "pr" in got
    assert "tas_alt" in got


def test_fallback_when_physical_missing_uses_branded(tmp_path: Path):
    """If Physical Parameter is empty, code falls back to Branded for match/return."""
    csv_path = _write_min_csv(tmp_path)
    got = find_variables_by_prefix(csv_path, "Amon.", exact=False)
    assert "name_from_branded" in got
