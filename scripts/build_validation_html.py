#!/usr/bin/env python3

"""Build a static HTML interface for browsing CMOR validation results.

This script scans ``<reports-dir>/*/validation_summary.json`` files written by
``validate_cmor_output.py`` and generates a self-contained static site at
``<reports-dir>/index.html``. Every validation run for another experiment,
realm or frequency adds a new subset directory; re-running this builder picks
it up, so the interface grows incrementally.

The site needs no web server: all report data is embedded in ``site_data.js``
and plot images are referenced with relative paths, so opening ``index.html``
directly (``file://``) works, and the whole ``validation_reports/`` folder can
be copied or synced as one unit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("cmip7_prep.build_validation_html")

DATA_FILENAME = "site_data.js"
INDEX_FILENAME = "index.html"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a static HTML interface from CMOR validation reports"
    )
    parser.add_argument(
        "--root-output-path",
        default=None,
        help="Root output directory containing validation_reports/",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory containing per-subset validation report folders "
        "(overrides <root-output-path>/validation_reports)",
    )
    parser.add_argument(
        "--html-dir",
        default=None,
        help="Directory to write the site to; defaults to the reports "
        "directory. When it differs, plot images are copied in so the "
        "site is self-contained (e.g. for a www-served directory).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def resolve_reports_dir(root_output_path: str | None, reports_dir: str | None) -> Path:
    """Resolve the validation reports directory from the CLI arguments."""
    if reports_dir:
        path = Path(reports_dir).expanduser().resolve()
    elif root_output_path:
        path = Path(root_output_path).expanduser().resolve() / "validation_reports"
    else:
        raise ValueError("Either --root-output-path or --reports-dir is required")
    if not path.is_dir():
        raise FileNotFoundError(f"Reports directory not found: {path}")
    return path


def collect_reports(reports_dir: Path) -> list[dict[str, Any]]:
    """Load every per-subset validation summary under the reports directory."""
    subsets = []
    for summary_path in sorted(reports_dir.glob("*/validation_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable report %s: %r", summary_path, exc)
            continue
        payload["subset_id"] = summary_path.parent.name
        subsets.append(payload)
    return subsets


def _plot_href(subset_prefix: str, plot_path: str | None) -> str | None:
    """Turn a report-relative plot path into an href relative to the site."""
    if not plot_path:
        return None
    if Path(plot_path).is_absolute():
        # Older reports stored absolute paths; these only render on the
        # machine that produced them but are kept rather than dropped.
        return plot_path
    return f"{subset_prefix}/{plot_path}"


def build_variable_rows(
    report: dict[str, Any], subset_prefix: str
) -> list[dict[str, Any]]:
    """Flatten one subset report into per-variable rows for the browse view."""
    expected = set(report.get("expected_variables", []))
    produced = set(report.get("produced_variables", []))
    missing = set(report.get("expected_but_not_produced", []))
    with_errors = set(report.get("variables_with_log_errors", []))
    provenance = report.get("variable_provenance", {})

    error_lines: dict[str, list[str]] = {}
    for record in report.get("log_records", []):
        if record.get("error_lines"):
            error_lines.setdefault(record["variable"], []).extend(record["error_lines"])

    inventory_by_short_name = {
        item["variable"]: item for item in report.get("dimension_inventory", [])
    }

    plots = report.get("plots", {})
    timeseries_plots = plots.get("timeseries", {})
    map_plots = plots.get("maps", {})
    zonal_plots = plots.get("zonal", {})
    # Reports from before per-variable plots stored lists; ignore those.
    if not isinstance(timeseries_plots, dict):
        timeseries_plots = {}
    if not isinstance(map_plots, dict):
        map_plots = {}
    if not isinstance(zonal_plots, dict):
        zonal_plots = {}

    rows = []
    for variable in sorted(expected | produced):
        short_name = variable.split("_")[0]
        inventory = inventory_by_short_name.get(short_name, {})
        zero_dims = sorted(
            {
                dim
                for sizes in inventory.get("sample_sizes", [])
                for dim, size in sizes.items()
                if size == 0
            }
        )
        if variable in missing:
            status = "missing"
        elif variable in with_errors:
            status = "error"
        elif zero_dims:
            status = "empty"
        else:
            status = "ok"
        rows.append(
            {
                "name": variable,
                "short_name": short_name,
                "status": status,
                "produced": variable in produced,
                "error_lines": error_lines.get(variable, [])[:10],
                "zero_dims": zero_dims,
                "dims": inventory.get("dims", []),
                "grid_types": inventory.get("grid_types", []),
                "sample_path": inventory.get("sample_path"),
                "provenance": provenance.get(variable),
                "timeseries_plot": _plot_href(
                    subset_prefix, timeseries_plots.get(variable)
                ),
                "map_plot": _plot_href(subset_prefix, map_plots.get(variable)),
                "zonal_plot": _plot_href(subset_prefix, zonal_plots.get(variable)),
            }
        )
    return rows


def build_site_data(subsets: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the data payload embedded into the static site."""
    site_subsets = []
    for report in subsets:
        subset_prefix = report["subset_id"]
        site_subsets.append(
            {
                "subset_id": report["subset_id"],
                "generated_at": report.get("generated_at"),
                "scope": report.get("scope", {}),
                "counts": report.get("counts", {}),
                "inspection_errors": report.get("inspection_errors", []),
                "variables": build_variable_rows(report, subset_prefix),
            }
        )
    return {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subsets": site_subsets,
    }


def write_site_data(data: dict[str, Any], output_path: Path) -> None:
    """Write the embedded data payload as a script file (works over file://)."""
    payload = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    output_path.write_text(
        f"window.CMOR_VALIDATION_DATA = {payload};\n", encoding="utf-8"
    )


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CMOR Validation</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #1c2733;
  --muted: #5b6b7b;
  --border: #d7dde4;
  --accent: #1f6feb;
  --ok: #1a7f37;
  --ok-bg: #dafbe1;
  --error: #cf222e;
  --error-bg: #ffebe9;
  --missing: #9a6700;
  --missing-bg: #fff8c5;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --text: #e6edf3;
    --muted: #8d96a0;
    --border: #30363d;
    --accent: #58a6ff;
    --ok: #3fb950;
    --ok-bg: #12261e;
    --error: #f85149;
    --error-bg: #2d1214;
    --missing: #d29922;
    --missing-bg: #272115;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding-block: 0 2rem;
  padding-inline: 16px;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
header {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 1rem;
  padding-block: 1rem 0.5rem;
}
header h1 { font-size: 1.3rem; margin: 0; }
header .meta { color: var(--muted); font-size: 0.85rem; }
nav.tabs { display: flex; gap: 0.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
nav.tabs button {
  border: none;
  background: none;
  color: var(--muted);
  font: inherit;
  padding: 0.5rem 1rem;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}
nav.tabs button.active { color: var(--text); border-bottom-color: var(--accent); font-weight: 600; }
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
}
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
tr:last-child td { border-bottom: none; }
.table-wrap { overflow-x: auto; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  display: inline-block;
  padding: 0.05rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
}
.badge.ok { color: var(--ok); background: var(--ok-bg); }
.badge.error { color: var(--error); background: var(--error-bg); }
.badge.missing { color: var(--missing); background: var(--missing-bg); }
.badge.empty { color: var(--missing); background: var(--missing-bg); outline: 1px dashed var(--missing); outline-offset: -1px; }
.badge.neutral { color: var(--muted); background: var(--bg); border: 1px solid var(--border); }
.bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--border); min-width: 120px; }
.bar span.ok { background: var(--ok); }
.bar span.error { background: var(--error); }
.bar span.missing { background: var(--missing); }
details.subset { margin-bottom: 0.75rem; }
details.subset > summary {
  cursor: pointer;
  font-weight: 600;
  padding: 0.25rem 0;
}
.issue-list { margin: 0.5rem 0 0.5rem 0; padding-left: 1.25rem; }
.issue-list li { margin-bottom: 0.35rem; }
.issue-list .reason { color: var(--muted); font-size: 0.85rem; display: block; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
.scopeline { color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; word-break: break-all; }
.filters { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
.filters label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.8rem; color: var(--muted); }
.filters select, .filters input {
  font: inherit;
  color: var(--text);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.35rem 0.5rem;
  min-width: 9rem;
}
.filters .count { margin-left: auto; color: var(--muted); align-self: center; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(420px, 100%), 1fr)); gap: 1rem; }
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.card .card-head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
.card .card-head .varname { font-weight: 600; word-break: break-all; }
.card .prov { color: var(--muted); font-size: 0.85rem; }
.card img { max-width: 100%; border: 1px solid var(--border); border-radius: 4px; background: #fff; }
.card .plots { display: flex; flex-direction: column; gap: 0.5rem; }
.card details summary { cursor: pointer; color: var(--muted); font-size: 0.85rem; }
.card .detail-body { font-size: 0.85rem; margin-top: 0.4rem; overflow-x: auto; }
.card .detail-body dt { color: var(--muted); float: left; clear: left; width: 7.5rem; }
.card .detail-body dd { margin-left: 8rem; word-break: break-all; }
.card .errbox {
  background: var(--error-bg);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  font-size: 0.82rem;
  overflow-x: auto;
}
.card .errbox div { white-space: pre-wrap; }
.empty { color: var(--muted); text-align: center; padding: 2rem; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<header>
  <h1>CMOR Validation</h1>
  <span class="meta" id="site-meta"></span>
</header>
<nav class="tabs">
  <button id="tab-overview" class="active">Overview</button>
  <button id="tab-browse">Browse</button>
</nav>

<section id="view-overview">
  <div class="panel table-wrap" id="overview-table"></div>
  <div id="overview-subsets"></div>
</section>

<section id="view-browse" hidden>
  <div class="panel filters" id="browse-filters"></div>
  <div class="cards" id="browse-cards"></div>
  <div class="empty" id="browse-empty" hidden>No variables match the current filters.</div>
</section>

<script src="site_data.js"></script>
<script>
(function () {
  "use strict";
  var DATA = window.CMOR_VALIDATION_DATA || { subsets: [] };
  var STATUS_LABELS = { ok: "OK", error: "Error", missing: "Missing", empty: "Empty" };

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") { node.textContent = attrs[key]; }
      else if (key === "html") { node.innerHTML = attrs[key]; }
      else { node.setAttribute(key, attrs[key]); }
    });
    (children || []).forEach(function (child) { node.appendChild(child); });
    return node;
  }

  function badge(status) {
    return el("span", { "class": "badge " + status, text: STATUS_LABELS[status] || status });
  }

  function subsetLabel(subset) {
    var scope = subset.scope || {};
    return [scope.model, scope.realm, scope.experiment, scope.frequency]
      .filter(Boolean).join(" · ") || subset.subset_id;
  }

  /* ---------------- Overview ---------------- */

  function progressBar(counts) {
    var expected = counts.expected_variables || 0;
    var errors = counts.variables_with_log_errors || 0;
    var missing = counts.missing_variables || 0;
    var ok = Math.max(expected - errors - missing, 0);
    var total = Math.max(expected, ok + errors + missing, 1);
    var bar = el("div", { "class": "bar", title: ok + " ok / " + errors + " errors / " + missing + " missing" });
    [["ok", ok], ["error", errors], ["missing", missing]].forEach(function (part) {
      if (part[1] > 0) {
        var span = el("span", { "class": part[0] });
        span.style.width = (100 * part[1] / total) + "%";
        bar.appendChild(span);
      }
    });
    return bar;
  }

  function renderOverviewTable() {
    var table = el("table");
    var head = el("tr");
    ["Subset", "Expected", "Produced", "Errors", "Missing", "Status", "Validated"].forEach(function (label) {
      head.appendChild(el("th", { text: label }));
    });
    table.appendChild(el("thead", {}, [head]));
    var body = el("tbody");
    DATA.subsets.forEach(function (subset) {
      var counts = subset.counts || {};
      var row = el("tr");
      row.appendChild(el("td", { text: subsetLabel(subset) }));
      row.appendChild(el("td", { "class": "num", text: counts.expected_variables != null ? counts.expected_variables : "–" }));
      row.appendChild(el("td", { "class": "num", text: counts.produced_variables != null ? counts.produced_variables : "–" }));
      row.appendChild(el("td", { "class": "num", text: counts.variables_with_log_errors != null ? counts.variables_with_log_errors : "–" }));
      row.appendChild(el("td", { "class": "num", text: counts.missing_variables != null ? counts.missing_variables : "–" }));
      row.appendChild(el("td", {}, [progressBar(counts)]));
      row.appendChild(el("td", { text: subset.generated_at || "–" }));
      body.appendChild(row);
    });
    table.appendChild(body);
    var wrap = document.getElementById("overview-table");
    wrap.textContent = "";
    if (DATA.subsets.length === 0) {
      wrap.appendChild(el("div", { "class": "empty", text: "No validation reports found yet. Run validate_cmor_output.py first." }));
    } else {
      wrap.appendChild(table);
    }
  }

  function issueItem(variable) {
    var item = el("li", {}, [el("span", { "class": "mono", text: variable.name })]);
    if (variable.error_lines.length > 0) {
      item.appendChild(el("span", { "class": "reason", text: variable.error_lines[0] }));
    }
    return item;
  }

  function renderOverviewSubsets() {
    var container = document.getElementById("overview-subsets");
    container.textContent = "";
    DATA.subsets.forEach(function (subset) {
      var scope = subset.scope || {};
      var errors = subset.variables.filter(function (v) { return v.status === "error"; });
      var missing = subset.variables.filter(function (v) { return v.status === "missing"; });
      var empty = subset.variables.filter(function (v) { return v.status === "empty"; });
      var body = el("div", { "class": "panel" });

      body.appendChild(el("div", { "class": "scopeline", text: "Output root: " + (scope.root_output_path || "unknown") }));
      body.appendChild(el("div", { "class": "scopeline", text: "Mapping YAML: " + (scope.yaml_path || "unknown") }));
      if (scope.ensemble_member) {
        body.appendChild(el("div", { "class": "scopeline", text: "Ensemble member: " + scope.ensemble_member }));
      }

      if (errors.length > 0) {
        body.appendChild(el("h3", { text: "Variables with CMOR log errors (" + errors.length + ")" }));
        body.appendChild(el("ul", { "class": "issue-list" }, errors.map(issueItem)));
      }
      if (missing.length > 0) {
        body.appendChild(el("h3", { text: "Expected but not produced (" + missing.length + ")" }));
        body.appendChild(el("ul", { "class": "issue-list" }, missing.map(function (v) {
          return el("li", {}, [el("span", { "class": "mono", text: v.name })]);
        })));
      }
      if (empty.length > 0) {
        body.appendChild(el("h3", { text: "Produced but with zero-length dimensions (" + empty.length + ")" }));
        body.appendChild(el("ul", { "class": "issue-list" }, empty.map(function (v) {
          var entry = el("li", {}, [el("span", { "class": "mono", text: v.name })]);
          entry.appendChild(el("span", { "class": "reason", text: "Zero-length dimension(s): " + v.zero_dims.join(", ") }));
          return entry;
        })));
      }
      if ((subset.inspection_errors || []).length > 0) {
        body.appendChild(el("h3", { text: "File inspection errors (" + subset.inspection_errors.length + ")" }));
        body.appendChild(el("ul", { "class": "issue-list" }, subset.inspection_errors.map(function (item) {
          var entry = el("li", {}, [el("span", { "class": "mono", text: item.variable || "?" })]);
          entry.appendChild(el("span", { "class": "reason", text: item.error || "" }));
          return entry;
        })));
      }
      if (errors.length === 0 && missing.length === 0 && empty.length === 0 && (subset.inspection_errors || []).length === 0) {
        body.appendChild(el("p", { text: "No problems found in this subset." }));
      }

      var summaryText = subsetLabel(subset) + " — " + errors.length + " errors, " + missing.length + " missing";
      if (empty.length > 0) { summaryText += ", " + empty.length + " empty"; }
      var details = el("details", { "class": "subset" }, [
        el("summary", { text: summaryText }),
        body
      ]);
      if (errors.length > 0 || missing.length > 0 || empty.length > 0) { details.setAttribute("open", ""); }
      container.appendChild(details);
    });
  }

  /* ---------------- Browse ---------------- */

  var filters = { model: "", experiment: "", frequency: "", realm: "", status: "", search: "" };

  function scopeValues(key) {
    var values = {};
    DATA.subsets.forEach(function (subset) {
      var value = (subset.scope || {})[key];
      if (value) { values[value] = true; }
    });
    return Object.keys(values).sort();
  }

  function renderFilters() {
    var container = document.getElementById("browse-filters");
    container.textContent = "";
    [["model", "Model"], ["experiment", "Experiment"], ["frequency", "Frequency"], ["realm", "Realm"]].forEach(function (pair) {
      var key = pair[0];
      var select = el("select");
      select.appendChild(el("option", { value: "", text: "All" }));
      scopeValues(key).forEach(function (value) {
        select.appendChild(el("option", { value: value, text: value }));
      });
      select.value = filters[key];
      select.addEventListener("change", function () { filters[key] = select.value; renderCards(); });
      container.appendChild(el("label", {}, [document.createTextNode(pair[1]), select]));
    });

    var statusSelect = el("select");
    [["", "All"], ["ok", "OK"], ["error", "Error"], ["missing", "Missing"], ["empty", "Empty"]].forEach(function (pair) {
      statusSelect.appendChild(el("option", { value: pair[0], text: pair[1] }));
    });
    statusSelect.value = filters.status;
    statusSelect.addEventListener("change", function () { filters.status = statusSelect.value; renderCards(); });
    container.appendChild(el("label", {}, [document.createTextNode("Status"), statusSelect]));

    var search = el("input", { type: "search", placeholder: "e.g. clt or tas_tavg" });
    search.value = filters.search;
    search.addEventListener("input", function () { filters.search = search.value.trim().toLowerCase(); renderCards(); });
    container.appendChild(el("label", {}, [document.createTextNode("Variable"), search]));

    container.appendChild(el("span", { "class": "count", id: "browse-count" }));
  }

  function matchingRows() {
    var rows = [];
    DATA.subsets.forEach(function (subset) {
      var scope = subset.scope || {};
      if (filters.model && scope.model !== filters.model) { return; }
      if (filters.experiment && scope.experiment !== filters.experiment) { return; }
      if (filters.frequency && scope.frequency !== filters.frequency) { return; }
      if (filters.realm && scope.realm !== filters.realm) { return; }
      subset.variables.forEach(function (variable) {
        if (filters.status && variable.status !== filters.status) { return; }
        if (filters.search && variable.name.toLowerCase().indexOf(filters.search) === -1) { return; }
        rows.push({ subset: subset, variable: variable });
      });
    });
    return rows;
  }

  function provenanceText(prov) {
    if (!prov) { return "No mapping entry found in YAML."; }
    var parts = [];
    if (prov.formula) {
      parts.push("Derived from " + prov.formula);
    } else if ((prov.source_model_vars || []).length > 0) {
      parts.push("From model variable " + prov.source_model_vars.join(", "));
    }
    if (prov.units) { parts.push("units: " + prov.units); }
    if (prov.regrid_method) { parts.push("regrid: " + prov.regrid_method); }
    return parts.join(" — ") || "No source information.";
  }

  function variableCard(subset, variable) {
    var card = el("div", { "class": "card" });
    var head = el("div", { "class": "card-head" }, [
      el("span", { "class": "varname", text: variable.name }),
      badge(variable.status),
      el("span", { "class": "badge neutral", text: subsetLabel(subset) })
    ]);
    card.appendChild(head);
    card.appendChild(el("div", { "class": "prov", text: provenanceText(variable.provenance) }));

    if (variable.error_lines.length > 0 || (variable.zero_dims || []).length > 0) {
      var errbox = el("div", { "class": "errbox" });
      (variable.zero_dims || []).forEach(function (dim) {
        errbox.appendChild(el("div", { text: "Produced file has zero-length dimension: " + dim }));
      });
      variable.error_lines.forEach(function (line) {
        errbox.appendChild(el("div", { text: line }));
      });
      card.appendChild(errbox);
    }

    var plots = el("div", { "class": "plots" });
    [["timeseries_plot", "Time series"], ["map_plot", "Time-mean map"], ["zonal_plot", "Zonal mean"]].forEach(function (pair) {
      var href = variable[pair[0]];
      if (!href) { return; }
      var img = el("img", { src: encodeURI(href), loading: "lazy", alt: pair[1] + " for " + variable.name });
      plots.appendChild(el("a", { href: encodeURI(href), target: "_blank", rel: "noopener" }, [img]));
    });
    if (plots.children.length > 0) { card.appendChild(plots); }
    else if (variable.produced) {
      card.appendChild(el("div", { "class": "prov", text: "No plots generated for this variable." }));
    }

    var dl = el("dl", {}, []);
    function detailRow(label, value) {
      if (!value || (Array.isArray(value) && value.length === 0)) { return; }
      dl.appendChild(el("dt", { text: label }));
      dl.appendChild(el("dd", { "class": "mono", text: Array.isArray(value) ? value.join("; ") : value }));
    }
    detailRow("Dimensions", variable.dims);
    detailRow("Grids", variable.grid_types);
    detailRow("Sample file", variable.sample_path);
    if (variable.provenance && variable.provenance.description) {
      detailRow("Description", variable.provenance.description);
    }
    if (dl.children.length > 0) {
      card.appendChild(el("details", {}, [
        el("summary", { text: "Details" }),
        el("div", { "class": "detail-body" }, [dl])
      ]));
    }
    return card;
  }

  function renderCards() {
    var rows = matchingRows();
    var container = document.getElementById("browse-cards");
    container.textContent = "";
    rows.forEach(function (row) {
      container.appendChild(variableCard(row.subset, row.variable));
    });
    document.getElementById("browse-empty").hidden = rows.length > 0;
    var count = document.getElementById("browse-count");
    if (count) { count.textContent = rows.length + " variable" + (rows.length === 1 ? "" : "s"); }
  }

  /* ---------------- Tabs and init ---------------- */

  function showTab(name) {
    document.getElementById("view-overview").hidden = name !== "overview";
    document.getElementById("view-browse").hidden = name !== "browse";
    document.getElementById("tab-overview").classList.toggle("active", name === "overview");
    document.getElementById("tab-browse").classList.toggle("active", name === "browse");
  }
  document.getElementById("tab-overview").addEventListener("click", function () { showTab("overview"); });
  document.getElementById("tab-browse").addEventListener("click", function () { showTab("browse"); });

  var totalVars = DATA.subsets.reduce(function (sum, subset) { return sum + subset.variables.length; }, 0);
  document.getElementById("site-meta").textContent =
    DATA.subsets.length + " validation subset" + (DATA.subsets.length === 1 ? "" : "s") +
    " · " + totalVars + " variables · built " + (DATA.built_at || "unknown");

  renderOverviewTable();
  renderOverviewSubsets();
  renderFilters();
  renderCards();
})();
</script>
</body>
</html>
"""


def sync_plots(report: dict[str, Any], reports_dir: Path, html_dir: Path) -> int:
    """Copy one subset's plot images into the HTML directory; returns the count copied.

    Only files that are missing or older at the destination are copied, so
    rebuilding an unchanged site is cheap.
    """
    subset_id = report["subset_id"]
    copied = 0
    for kind_paths in (report.get("plots") or {}).values():
        if not isinstance(kind_paths, dict):
            continue
        for plot_path in kind_paths.values():
            if not plot_path or Path(plot_path).is_absolute():
                continue
            source = reports_dir / subset_id / plot_path
            dest = html_dir / subset_id / plot_path
            if not source.is_file():
                logger.warning("Referenced plot not found: %s", source)
                continue
            if dest.is_file() and dest.stat().st_mtime >= source.stat().st_mtime:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied += 1
    return copied


def build_site(reports_dir: str | Path, html_dir: str | Path | None = None) -> Path:
    """Build (or rebuild) the static site; returns the path to index.html."""
    reports_dir = Path(reports_dir).expanduser().resolve()
    html_dir = Path(html_dir).expanduser().resolve() if html_dir else reports_dir
    html_dir.mkdir(parents=True, exist_ok=True)
    subsets = collect_reports(reports_dir)
    if not subsets:
        logger.warning("No validation_summary.json files found under %s", reports_dir)
    if html_dir != reports_dir:
        copied = sum(sync_plots(report, reports_dir, html_dir) for report in subsets)
        logger.info("Copied %d plot file(s) into %s", copied, html_dir)
    data = build_site_data(subsets)
    write_site_data(data, html_dir / DATA_FILENAME)
    index_path = html_dir / INDEX_FILENAME
    index_path.write_text(INDEX_HTML, encoding="utf-8")
    logger.info(
        "Built HTML interface with %d subset(s) at %s", len(subsets), index_path
    )
    return index_path


def main() -> int:
    """Run the HTML builder from the command line."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    reports_dir = resolve_reports_dir(args.root_output_path, args.reports_dir)
    index_path = build_site(reports_dir, args.html_dir)
    print(f"HTML interface written to {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
