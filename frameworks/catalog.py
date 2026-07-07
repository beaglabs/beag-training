"""Framework catalog data models and loader.

All three NIST frameworks are represented with a unified data model:
  - 800-53 Rev 5: 20 families → controls → enhancements
  - CSF 2.0:       6 functions → 22 categories → 106 subcategories
  - CMMC 2.0:     14 domains → capabilities → CMMC practices

Every entry carries cross-framework mappings that are populated by the
crosswalk module after all frameworks have been loaded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


class Framework(str, Enum):
    NIST_800_53 = "nist_800_53"
    NIST_CSF = "nist_csf"
    CMMC = "cmmc"


class Granularity(str, Enum):
    FAMILY = "family"
    CONTROL = "control"
    ENHANCEMENT = "enhancement"


@dataclass
class CrosswalkRef:
    """A single cross-framework reference."""

    framework: Framework
    control_id: str
    label: str = ""
    relationship: str = "maps-to"


@dataclass
class ControlEntry:
    """A single control / subcategory / practice within a framework.

    The hierarchy is modelled via ``parent_id`` and ``children``, so the
    same type represents family-level, control-level, and enhancement-level
    entries.
    """

    id: str
    title: str
    description: str
    framework: Framework
    granularity: Granularity
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    crosswalk: list[CrosswalkRef] = field(default_factory=list)
    supplemental_guidance: str = ""

    def text_for_embedding(self) -> str:
        """Composite text used for vector-search embeddings."""
        parts = [self.title, self.description]
        if self.supplemental_guidance:
            parts.append(self.supplemental_guidance)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "framework": self.framework.value,
            "granularity": self.granularity.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "crosswalk": [
                {
                    "framework": cw.framework.value,
                    "control_id": cw.control_id,
                    "label": cw.label,
                    "relationship": cw.relationship,
                }
                for cw in self.crosswalk
            ],
            "supplemental_guidance": self.supplemental_guidance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlEntry:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            framework=Framework(data["framework"]),
            granularity=Granularity(data["granularity"]),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            crosswalk=[
                CrosswalkRef(
                    framework=Framework(cw["framework"]),
                    control_id=cw["control_id"],
                    label=cw.get("label", ""),
                    relationship=cw.get("relationship", "maps-to"),
                )
                for cw in data.get("crosswalk", [])
            ],
            supplemental_guidance=data.get("supplemental_guidance", ""),
        )


@dataclass
class Catalog:
    frameworks: dict[Framework, list[ControlEntry]] = field(default_factory=dict)
    _by_id: dict[tuple[Framework, str], ControlEntry] = field(default_factory=dict)

    def add(self, entry: ControlEntry) -> None:
        self.frameworks.setdefault(entry.framework, []).append(entry)
        self._by_id[(entry.framework, entry.id)] = entry

    def get(self, framework: Framework, control_id: str) -> ControlEntry | None:
        return self._by_id.get((framework, control_id))

    def list_by_framework(
        self, framework: Framework, granularity: Granularity | None = None
    ) -> list[ControlEntry]:
        entries = self.frameworks.get(framework, [])
        if granularity is not None:
            entries = [e for e in entries if e.granularity == granularity]
        return entries

    def search_text(self, query: str) -> list[ControlEntry]:
        q = query.lower()
        results: list[ControlEntry] = []
        for fw_entries in self.frameworks.values():
            for entry in fw_entries:
                if q in entry.title.lower() or q in entry.description.lower():
                    results.append(entry)
        return results

    def all_entries(self) -> list[ControlEntry]:
        return [e for entries in self.frameworks.values() for e in entries]

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for fw, entries in self.frameworks.items():
            out[fw.value] = len(entries)
        out["total"] = sum(len(v) for v in self.frameworks.values())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "frameworks": {
                fw.value: [e.to_dict() for e in entries]
                for fw, entries in self.frameworks.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalog:
        cat = cls()
        for fw_value, entries in data.get("frameworks", {}).items():
            for e_data in entries:
                cat.add(ControlEntry.from_dict(e_data))
        return cat

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path else DEFAULT_CATALOG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> Catalog:
        target = Path(path) if path else DEFAULT_CATALOG_PATH
        if not target.exists():
            raise FileNotFoundError(f"Catalog not found at {target}. Run `build` first.")
        return cls.from_dict(json.loads(target.read_text()))


def load_catalog(path: Path | str | None = None, force_rebuild: bool = False) -> Catalog:
    """Load the catalog from disk, optionally rebuilding first."""
    target = Path(path) if path else DEFAULT_CATALOG_PATH

    if not force_rebuild and target.exists():
        return Catalog.load(target)

    from frameworks.fetchers.nist_800_53 import fetch_800_53
    from frameworks.fetchers.nist_csf import fetch_csf
    from frameworks.fetchers.cmmc import fetch_cmmc

    cat = Catalog()
    for entry in fetch_800_53():
        cat.add(entry)
    for entry in fetch_csf():
        cat.add(entry)
    for entry in fetch_cmmc():
        cat.add(entry)

    from frameworks.crosswalk import populate_crosswalks

    populate_crosswalks(cat)
    cat.save(target)
    return cat
