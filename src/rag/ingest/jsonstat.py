"""Generic JSON-stat 2.0 parser.

Turns a JSON-stat dataset (as returned by the CSO PxStat API) into a flat list
of observation records, plus dataset-level metadata. The parser is generic: it
works for any matrix regardless of its dimensions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any

_YEAR_RE = re.compile(r"(19|20)\d{2}")


@dataclass
class DatasetMeta:
    matrix: str
    label: str
    source: str
    copyright: str
    updated: str
    href: str
    time_dim: str | None


@dataclass
class Observation:
    """A single statistic value with its dimension labels."""

    matrix: str
    statistic: str
    value: float
    unit: str
    dims: dict[str, str] = field(default_factory=dict)  # dim_label -> category_label
    year: int | None = None


def _category_order(dim: dict[str, Any]) -> list[str]:
    """Return category keys in their declared order."""
    cat = dim["category"]
    index = cat["index"]
    if isinstance(index, dict):
        # index may be {key: position}; sort by position
        return [k for k, _ in sorted(index.items(), key=lambda kv: kv[1])]
    return list(index)


def parse_dataset(doc: dict[str, Any]) -> tuple[DatasetMeta, list[Observation]]:
    """Parse a JSON-stat 2.0 document into (metadata, observations)."""
    if doc.get("class") != "dataset":
        raise ValueError("Not a JSON-stat dataset document")

    ext = doc.get("extension", {})
    matrix = ext.get("matrix") or doc.get("label", "UNKNOWN")
    copyright_name = (ext.get("copyright") or {}).get("name", "")
    role = doc.get("role", {}) or {}
    time_dims = role.get("time", []) or []
    metric_dims = role.get("metric", []) or []
    time_dim_id = time_dims[0] if time_dims else None

    meta = DatasetMeta(
        matrix=matrix,
        label=doc.get("label", matrix),
        source=copyright_name or "Central Statistics Office, Ireland (CSO)",
        copyright=copyright_name,
        updated=doc.get("updated", ""),
        href=doc.get("href", ""),
        time_dim=time_dim_id,
    )

    dim_ids: list[str] = doc["id"]
    dimension = doc["dimension"]
    values = doc["value"]

    # Pre-compute, per dimension: ordered category keys, key->label, label string.
    ordered_keys: dict[str, list[str]] = {}
    labels: dict[str, dict[str, str]] = {}
    dim_label: dict[str, str] = {}
    units: dict[str, dict[str, dict]] = {}
    for did in dim_ids:
        d = dimension[did]
        ordered_keys[did] = _category_order(d)
        labels[did] = d["category"].get("label", {})
        dim_label[did] = d.get("label", did)
        units[did] = d["category"].get("unit", {})

    metric_id = metric_dims[0] if metric_dims else dim_ids[0]

    observations: list[Observation] = []
    # value array is row-major over dim_ids order.
    key_lists = [ordered_keys[d] for d in dim_ids]
    for flat_idx, combo in enumerate(product(*key_lists)):
        v = values[flat_idx] if flat_idx < len(values) else None
        if v is None:
            continue
        try:
            fval = float(v)
        except (TypeError, ValueError):
            continue

        dim_map: dict[str, str] = {}
        statistic = ""
        unit = ""
        year: int | None = None
        for did, key in zip(dim_ids, combo):
            lab = labels[did].get(key, key)
            dim_map[dim_label[did]] = lab
            if did == metric_id:
                statistic = lab
                u = units[did].get(key, {})
                unit = u.get("label", "") if isinstance(u, dict) else ""
            if did == time_dim_id:
                m = _YEAR_RE.search(str(lab))
                if m:
                    year = int(m.group(0))

        observations.append(
            Observation(
                matrix=matrix,
                statistic=statistic or "value",
                value=fval,
                unit=unit,
                dims=dim_map,
                year=year,
            )
        )
    return meta, observations
