"""Data augmentation for synthetic training examples.

Techniques:
  - Control swapping: replace a mapped control with an adjacent control
    (same family) to create negative examples and force the model to
    learn fine-grained distinctions.
  - Noise injection: add irrelevant JSON keys, remove keys, corrupt values
  - Ambiguity amplification: add control references that are plausible but
    incorrect to force multi-label reasoning
"""

from __future__ import annotations

import copy
import random
from typing import Any

from frameworks.catalog import Catalog, Framework
from model.output_schema import ControlMapping, TrainingExample


class DataAugmenter:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def augment(
        self,
        example: TrainingExample,
        techniques: list[str] | None = None,
    ) -> list[TrainingExample]:
        if techniques is None:
            techniques = ["swap_control", "noise_keys", "drop_keys"]
        results: list[TrainingExample] = []
        for tech in techniques:
            if tech == "swap_control" and len(example.mappings) > 0:
                aug = self._swap_control(example)
                if aug is not None:
                    results.append(aug)
            elif tech == "noise_keys":
                results.append(self._inject_noise(example))
            elif tech == "drop_keys":
                results.append(self._drop_keys(example))
        return results

    # ------------------------------------------------------------------
    # Techniques
    # ------------------------------------------------------------------

    def _swap_control(self, example: TrainingExample) -> TrainingExample | None:
        """Replace one mapping with a closely-related but incorrect control."""
        ex = copy.deepcopy(example)
        idx = random.randrange(len(ex.mappings))
        original = ex.mappings[idx]

        fw = Framework(str(original.framework)) if original.framework else Framework.NIST_800_53
        controls = [
            e for e in self.catalog.list_by_framework(fw)
            if e.granularity.value in ("control", "enhancement")
            and e.id != original.control_id
        ]
        if not controls:
            return None

        alt = random.choice(controls)
        ex.mappings[idx] = ControlMapping(
            control_id=alt.id,
            control_title=alt.title,
            framework=str(fw.value),
            granularity=str(alt.granularity.value),
            confidence=1.0,
            reasoning=ex.mappings[idx].reasoning,
        )
        return ex

    def _inject_noise(self, example: TrainingExample) -> TrainingExample:
        ex = copy.deepcopy(example)
        noise_keys = [
            "timestamp", "version", "internal_id", "reviewed_by",
            "department_code", "source_system", "metadata", "tags",
        ]
        for _ in range(random.randint(1, 3)):
            key = random.choice(noise_keys)
            if isinstance(ex.input_json, dict) and key not in ex.input_json:
                ex.input_json[key] = f"noise_{random.randint(1000, 9999)}"  # type: ignore[index]
        return ex

    def _drop_keys(self, example: TrainingExample) -> TrainingExample:
        ex = copy.deepcopy(example)
        if isinstance(ex.input_json, dict) and len(ex.input_json) > 2:
            keys = list(ex.input_json.keys())
            essential = {"document_type", "title", "finding_id", "risk_id", "gap_id", "implementation_id"}
            droppable = [k for k in keys if k not in essential]
            if droppable:
                key = random.choice(droppable)
                del ex.input_json[key]  # type: ignore[arg-type]
        return ex
