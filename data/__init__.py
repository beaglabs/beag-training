from __future__ import annotations

import json
from typing import Any

from data.generator import DataGenerator, GeneratorConfig
from data.augment import DataAugmenter
from data.validator import validate_example, filter_valid

__all__ = [
    "DataAugmenter",
    "DataGenerator",
    "GeneratorConfig",
    "as_instruction_messages",
    "filter_valid",
    "records_to_instructions",
    "validate_example",
]

SYSTEM_PROMPT = (
    "You are a NIST compliance expert. Map the given document to relevant "
    "NIST 800-53, CSF, and CMMC controls. Return a JSON array of control "
    "mappings with control_id, framework, confidence, and reasoning."
)


def as_instruction_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a record with 'text' + 'mappings' to chat messages.

    The text field contains the document JSON, mappings is a list of
    control assignment dicts. Returns [system, user, assistant] messages.
    """
    doc = record.get("text", "")
    if isinstance(doc, dict):
        doc = json.dumps(doc)
    mappings = record.get("mappings", [])
    assistant = json.dumps(mappings)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": doc},
        {"role": "assistant", "content": assistant},
    ]


def records_to_instructions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of records to instruction format, preserving metadata."""
    out = []
    for rec in records:
        new_rec = {
            "messages": as_instruction_messages(rec),
            "mappings": rec.get("mappings", []),
            "text_context": rec.get("text_context", ""),
        }
        for k, v in rec.items():
            if k not in ("text", "input_json", "messages", "mappings"):
                new_rec[k] = v
        out.append(new_rec)
    return out
