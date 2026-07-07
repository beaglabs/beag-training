"""Structured output format for the NIST compliance classification model.

The model produces JSON with a ``mappings`` array where each mapping includes
the target control, framework, granularity, confidence score, and a short
reasoning trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frameworks.catalog import Framework, Granularity


@dataclass
class ControlMapping:
    control_id: str
    control_title: str = ""
    framework: Framework | str = ""
    granularity: Granularity | str = ""
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "control_title": self.control_title,
            "framework": self.framework.value if isinstance(self.framework, Framework) else self.framework,
            "granularity": self.granularity.value if isinstance(self.granularity, Granularity) else self.granularity,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class ClassificationResult:
    mappings: list[ControlMapping] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"mappings": [m.to_dict() for m in self.mappings]}


@dataclass
class TrainingExample:
    input_json: dict[str, Any]
    mappings: list[ControlMapping] = field(default_factory=list)
    text_context: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {
            "input_json": self.input_json,
            "mappings": [m.to_dict() for m in self.mappings],
            "text_context": self.text_context,
        }

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> TrainingExample:
        mappings = []
        for m in data.get("mappings", []):
            mappings.append(
                ControlMapping(
                    control_id=m["control_id"],
                    control_title=m.get("control_title", ""),
                    framework=m.get("framework", ""),
                    granularity=m.get("granularity", ""),
                    confidence=m.get("confidence", 1.0),
                    reasoning=m.get("reasoning", ""),
                )
            )
        return cls(
            input_json=data.get("input_json", {}),
            mappings=mappings,
            text_context=data.get("text_context", ""),
        )
