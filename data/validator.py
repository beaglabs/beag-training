"""Validation for generated training examples.

Validates:
  - JSON structure and required fields
  - Control IDs reference real entries in the framework catalog
  - Framework identifiers are valid
  - Granularity values are valid
"""

from __future__ import annotations

from typing import Sequence

from frameworks.catalog import Catalog
from model.output_schema import TrainingExample


def validate_example(example: TrainingExample, catalog: Catalog) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(example.input_json, dict):
        errors.append("input_json is not a dict")
    if not example.mappings:
        errors.append("no mappings found")

    for i, m in enumerate(example.mappings):
        if not m.control_id:
            errors.append(f"mapping {i}: missing control_id")
            continue

        try:
            from frameworks.catalog import Framework as Fw
            fw = Fw(str(m.framework)) if m.framework else Fw.NIST_800_53
        except ValueError:
            errors.append(f"mapping {i}: invalid framework '{m.framework}'")
            continue

        entry = catalog.get(fw, m.control_id)
        if entry is None:
            errors.append(f"mapping {i}: unknown control '{fw.value}:{m.control_id}'")

        try:
            from frameworks.catalog import Granularity
            Granularity(str(m.granularity))
        except ValueError:
            errors.append(f"mapping {i}: invalid granularity '{m.granularity}'")

        if not (0.0 <= m.confidence <= 1.0):
            errors.append(f"mapping {i}: confidence out of range ({m.confidence})")

    return len(errors) == 0, errors


def filter_valid(
    examples: Sequence[TrainingExample],
    catalog: Catalog,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    valid: list[TrainingExample] = []
    invalid: list[TrainingExample] = []
    for ex in examples:
        ok, _ = validate_example(ex, catalog)
        if ok:
            valid.append(ex)
        else:
            invalid.append(ex)
    return valid, invalid
