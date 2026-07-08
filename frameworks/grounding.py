"""Catalog-grounded control ID validation and correction.

After the model generates control mappings, this module validates each
control_id against the framework catalog and corrects hallucinated IDs
by fuzzy-matching against valid catalog entries.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from frameworks.catalog import Catalog, ControlEntry, Framework


def build_id_index(cat: Catalog) -> dict[str, list[ControlEntry]]:
    """Build a case-insensitive index of control IDs to catalog entries."""
    index: dict[str, list[ControlEntry]] = {}
    for entry in cat.all_entries():
        key = _normalize(entry.id)
        index.setdefault(key, []).append(entry)
    return index


def validate_mapping(
    mapping: dict[str, Any],
    id_index: dict[str, list[ControlEntry]],
) -> dict[str, Any]:
    """Validate a single mapping dict. Returns corrected version.

    If control_id is valid, returns mapping unchanged with validated=True.
    If control_id is hallucinated, attempts fuzzy match and returns
    the best correction with validated=False.
    """
    cid = mapping.get("control_id", "")
    framework_str = mapping.get("framework", "")

    # Direct lookups
    normalized = _normalize(cid)
    entries = id_index.get(normalized, [])
    if entries:
        return {**mapping, "validated": True, "canonical_entry": entries[0].id}

    # Try by framework
    fw = _parse_framework(framework_str)
    if fw and entries_from_fw(id_index, normalized, fw):
        best = entries_from_fw(id_index, normalized, fw)[0]
        return {**mapping, "control_id": best.id, "control_title": best.title,
                "framework": best.framework.value, "validated": False, "corrected": True}

    # Fuzzy match across all entries
    best = _fuzzy_match(cid, id_index, framework_str)
    if best and best[1] > 0.5:
        entry = best[0]
        return {**mapping, "control_id": entry.id, "control_title": entry.title,
                "framework": entry.framework.value, "validated": False, "corrected": True}

    return {**mapping, "validated": False, "corrected": False,
            "warning": "No matching control found in catalog"}


def ground_control_ids(
    mappings: list[dict[str, Any]],
    cat: Catalog,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and correct all mappings against the catalog.

    Returns (corrected_mappings, report) where report has
    validation statistics.
    """
    id_index = build_id_index(cat)
    corrected: list[dict[str, Any]] = []
    stats = {"total": len(mappings), "valid": 0, "corrected": 0, "dropped": 0}

    for m in mappings:
        result = validate_mapping(m, id_index)
        if result.get("validated"):
            stats["valid"] += 1
        elif result.get("corrected"):
            stats["corrected"] += 1
        else:
            stats["dropped"] += 1
        corrected.append(result)

    return corrected, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "-")


def _parse_framework(framework_str: str) -> Framework | None:
    fw_map = {
        "nist_800_53": Framework.NIST_800_53,
        "nist 800-53": Framework.NIST_800_53,
        "800-53": Framework.NIST_800_53,
        "nist_csf": Framework.NIST_CSF,
        "csf": Framework.NIST_CSF,
        "nist csf": Framework.NIST_CSF,
        "cmmc": Framework.CMMC,
    }
    return fw_map.get(_normalize(framework_str))


def entries_from_fw(
    id_index: dict[str, list[ControlEntry]], cid: str, fw: Framework
) -> list[ControlEntry]:
    return [e for e in id_index.get(cid, []) if e.framework == fw]


def _fuzzy_match(
    cid: str,
    id_index: dict[str, list[ControlEntry]],
    framework_str: str = "",
) -> tuple[ControlEntry, float] | None:
    """Find the closest control ID in the catalog by fuzzy string matching."""
    fw = _parse_framework(framework_str)
    candidates = []
    for entries in id_index.values():
        for e in entries:
            if fw and e.framework != fw:
                continue
            candidates.append(e)

    if not candidates:
        return None

    best_entry = None
    best_score = 0.0
    target = _normalize(cid)

    for e in candidates:
        score = SequenceMatcher(None, target, _normalize(e.id)).ratio()
        # Boost score for entries that share a prefix
        if target[:3] == _normalize(e.id)[:3]:
            score += 0.15
        if score > best_score:
            best_score = score
            best_entry = e

    if best_entry:
        return best_entry, best_score
    return None
