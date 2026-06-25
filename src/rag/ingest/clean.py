"""Clean & transform parsed observations into citable text documents.

CSO data is tabular. Raw numbers are poor for text retrieval, so we convert
each statistical series into a compact natural-language document:

    "Consumer Price Index (Base Dec 2023=100) — All items — Ireland (CSO,
     matrix CPM01). Annual values: 2015: 89.6; 2016: 89.9; ... ."

Each document carries citation metadata (matrix, statistic, source, URL).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from rag.ingest.jsonstat import DatasetMeta, Observation
from rag.logging_utils import get_logger

log = get_logger(__name__)

_CSO_DATASET_URL = "https://data.cso.ie/table/{matrix}"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:80]


def _fmt(value: float, unit: str) -> str:
    # Integer-valued counts: render with thousands separators (no sci notation).
    if float(value).is_integer():
        s = f"{int(value):,}"
    else:
        s = f"{value:g}"
    if unit and unit not in ("", "Number"):
        if unit == "%":
            return f"{s}%"
        return f"{s} ({unit})"
    return s


# Domain alias phrases: make tabular series retrievable by natural questions.
_ALIASES = [
    (("consumer price", "cpi"), "This series measures consumer price inflation, "
     "the inflation rate and the cost of living in Ireland."),
    (("unemployment", "labour", "jobless"), "This series measures the unemployment "
     "rate (jobless rate) in the Irish labour market."),
    (("population", "census"), "This series measures the population of Ireland — "
     "how many people live in the country."),
    (("earnings", "wage", "income"), "This series measures earnings, wages and "
     "income in Ireland."),
]


def _aliases_for(text: str) -> str:
    low = text.lower()
    out = []
    for keys, phrase in _ALIASES:
        if any(k in low for k in keys):
            out.append(phrase)
    return " ".join(out)


def observations_to_documents(
    meta: DatasetMeta,
    observations: Iterable[Observation],
    min_year: int | None = None,
) -> list[Document]:
    """Group observations into one document per (statistic, non-time dims)."""
    time_label = None  # the human dimension name used as time, discovered below
    groups: dict[tuple, dict] = {}

    obs_list = list(observations)
    # Identify the time dimension's human label by matching to year presence.
    for o in obs_list:
        if o.year is not None:
            for dname, dval in o.dims.items():
                if str(o.year) in str(dval):
                    time_label = dname
                    break
        if time_label:
            break

    kept = 0
    for o in obs_list:
        if min_year is not None and (o.year is None or o.year < min_year):
            continue
        non_time = {k: v for k, v in o.dims.items() if k != time_label}
        # Drop the statistic dim from the descriptor (it's the metric itself).
        descriptor = {k: v for k, v in non_time.items() if v != o.statistic}
        gkey = (o.statistic, tuple(sorted(descriptor.items())))
        g = groups.setdefault(
            gkey,
            {"statistic": o.statistic, "unit": o.unit, "descriptor": descriptor, "years": {}},
        )
        # last observation for a given year wins (most recent month)
        g["years"][o.year] = o.value
        kept += 1

    log.info(
        "matrix %s: %d observations -> %d series (min_year=%s)",
        meta.matrix, kept, len(groups), min_year,
    )

    docs: list[Document] = []
    url = _CSO_DATASET_URL.format(matrix=meta.matrix)
    for (statistic, _), g in groups.items():
        desc = g["descriptor"]
        desc_str = ", ".join(f"{k}: {v}" for k, v in desc.items())
        title_bits = [statistic]
        if desc_str:
            title_bits.append(desc_str)
        title = f"{meta.label} — " + " — ".join(title_bits)

        years = sorted(g["years"].items())
        if not years:
            continue
        value_sentences = "; ".join(
            f"{yr}: {_fmt(val, g['unit'])}" for yr, val in years
        )
        latest_yr, latest_val = years[-1]
        body = (
            f"{statistic}"
            + (f" ({desc_str})" if desc_str else "")
            + " for Ireland, from the Central Statistics Office dataset "
            + f"{meta.label} (matrix {meta.matrix}). "
            + f"Most recent value ({latest_yr}): {_fmt(latest_val, g['unit'])}. "
            + f"Annual values — {value_sentences}. "
            + (_aliases_for(f"{meta.label} {statistic} {desc_str}") + " ")
            + "Source: Central Statistics Office, Ireland (CSO), CC BY 4.0."
        )
        doc_id = _slug(f"{meta.matrix}-{statistic}-{desc_str}")
        docs.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=body,
                metadata={
                    "matrix": meta.matrix,
                    "dataset": meta.label,
                    "statistic": statistic,
                    "descriptor": desc_str,
                    "unit": g["unit"],
                    "source": meta.source,
                    "license": "CC BY 4.0",
                    "url": url,
                    "updated": meta.updated,
                    "latest_year": latest_yr,
                },
            )
        )
    return docs


def deduplicate(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        norm = re.sub(r"\s+", " ", d.text).strip().lower()
        h = str(hash(norm))
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    if len(out) != len(docs):
        log.info("deduplicated %d -> %d documents", len(docs), len(out))
    return out
