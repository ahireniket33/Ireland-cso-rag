"""Build small, valid JSON-stat 2.0 sample fixtures from verified CSO headline
figures, so the pipeline and tests run fully offline.

These are a SMALL illustrative extract. The live ingestion pipeline
(`python run.py ingest`) pulls the complete, authoritative series from the CSO
PxStat API. Figures here are real CSO headline numbers (sources in
DATA_LICENSES.md); full monthly detail is omitted for size.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)


def build(matrix: str, label: str, statistic: str, unit_label: str,
          time_dim: str, year_values: dict[int, float], updated: str) -> dict:
    years = sorted(year_values)
    time_keys = [str(y) for y in years]
    doc = {
        "class": "dataset",
        "label": label,
        "id": ["STATISTIC", time_dim],
        "size": [1, len(years)],
        "role": {"metric": ["STATISTIC"], "time": [time_dim]},
        "dimension": {
            "STATISTIC": {
                "label": "Statistic",
                "category": {
                    "index": ["S1"],
                    "label": {"S1": statistic},
                    "unit": {"S1": {"label": unit_label, "decimals": 1}},
                },
            },
            time_dim: {
                "label": "Year",
                "category": {
                    "index": time_keys,
                    "label": {k: k for k in time_keys},
                },
            },
        },
        "value": [year_values[y] for y in years],
        "updated": updated,
        "href": f"https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{matrix}/JSON-stat/2.0/en",
        "extension": {
            "matrix": matrix,
            "copyright": {"name": "Central Statistics Office, Ireland", "code": "CSO",
                          "href": "https://www.cso.ie/"},
        },
    }
    (OUT / f"{matrix}.jsonstat.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


# --- CPM01: CPI annual % change (All items) — verified headline figures -------
build(
    "CPM01", "Consumer Price Index",
    "Percentage Change over 12 months for Consumer Price Index (All items)", "%",
    "TLIST(A1)",
    {2015: -0.3, 2016: 0.0, 2017: 0.4, 2018: 0.5, 2019: 0.9,
     2020: -0.3, 2021: 2.4, 2022: 7.8, 2023: 6.3, 2024: 2.1, 2025: 2.2},
    "2026-01-15T11:00:00.000Z",
)

# --- MUM01: Unemployment rate (15-74), both sexes, annual ---------------------
build(
    "MUM01", "Monthly Unemployment",
    "Seasonally Adjusted Unemployment Rate (15-74 years), Both sexes", "%",
    "TLIST(A1)",
    {2015: 9.9, 2016: 8.4, 2017: 6.7, 2018: 5.7, 2019: 5.0,
     2020: 5.8, 2021: 6.2, 2022: 4.5, 2023: 4.3, 2024: 4.0},
    "2026-01-08T11:00:00.000Z",
)

# --- FY001A: Census total population by census year ---------------------------
build(
    "FY001A", "Census Population",
    "Population (Both sexes, All ages)", "Number",
    "TLIST(A1)",
    {2006: 4239848, 2011: 4588252, 2016: 4761865, 2022: 5149139},
    "2023-05-30T11:00:00.000Z",
)

print("Sample fixtures written to", OUT)
for p in sorted(OUT.glob("*.json")):
    print(" -", p.name, p.stat().st_size, "bytes")
