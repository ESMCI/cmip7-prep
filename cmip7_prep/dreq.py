# cmip7_prep/dreq.py
import json, csv
from dataclasses import dataclass

@dataclass
class VarDef:
    name: str
    realm: str
    units: str
    dimensions: list[str]
    cell_methods: str | None
    positive: str | None
    long_name: str
    freq: str
    table: str
    requires_plev19: bool
    regrid_method: str  # 'conservative' or 'bilinear' default rules

class DReq:
    def __init__(self, export_path, mapping_yaml):
        self.vars = self._load(export_path)
        self.mapping = self._load_yaml(mapping_yaml)

    def _load(self, path):
        if str(path).endswith(".json"):
            data = json.load(open(path))
            # normalize to a dict keyed by (realm, var)
        else:
            # parse CSV export from Airtable "Variables" tab (MASTER view)
            data = list(csv.DictReader(open(path, newline="")))
        # ...extract fields like variable compound name (e.g., Amon.tas), units, dims, cell methods...
        return self._normalize(data)

    def _normalize(self, rows):
        out = {}
        for r in rows:
            compound = r.get("Compound name", "")  # e.g., "Amon.tas"
            if "." in compound:
                realm, vname = compound.split(".", 1)
            else:
                realm, vname = r.get("Table",""), r.get("Short name","")
            v = VarDef(
                name=vname,
                realm=realm,
                units=r.get("Units",""),
                dimensions=r.get("Dimensions","").split(),
                cell_methods=r.get("Cell methods") or None,
                positive=r.get("Positive") or None,
                long_name=r.get("Long name") or vname,
                freq=r.get("Frequency",""),
                table=realm,
                requires_plev19=("plev19" in r.get("Dimensions","")),
                regrid_method=self._choose_method(vname, r),
            )
            out[(realm, vname)] = v
        return out

    def _choose_method(self, vname, r):
        # crude default: flux/accumulations → conservative; else bilinear/patch
        cm = (r.get("Cell methods") or "").lower()
        if any(k in cm for k in ["sum", "time: sum"]) or vname in {"pr","evspsbl","hfss","hfls"}:
            return "conservative_normed"
        return "bilinear"

    def lookup(self, realm, var):
        v = self.vars.get((realm, var))
        # patch with mapping yaml if present (e.g., units overrides, name aliases)
        return v

    def dataset_attrs(self):
        # Return global attributes for CMOR dataset (institution_id, source_id, etc.)
        return {}

