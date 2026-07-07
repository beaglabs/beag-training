"""Prompt templates for generating synthetic JSON → NIST mapping examples.

Each template defines a JSON document type (policy, audit finding, risk
assessment, control implementation, gap analysis) and instructs the frontier
model to generate plausible JSON that maps to specified NIST controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from frameworks.catalog import ControlEntry, Framework

DocType = Literal["policy", "finding", "risk", "implementation", "gap"]


@dataclass
class GenerationTemplate:
    system: str
    user_prefix: str

    def build_messages(
        self,
        controls: list[ControlEntry],
        doc_type: DocType,
        num_mappings: int = 3,
    ) -> list[dict[str, str]]:
        ctx = _build_control_context(controls)
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_prefix.format(
                doc_type=doc_type,
                num_mappings=num_mappings,
                controls=ctx,
            )},
        ]


# ── Policy document ─────────────────────────────────────────────────

POLICY_SYSTEM = """You are a compliance expert generating realistic corporate policy documents.

Your task: create a plausible JSON policy document that would map to specific
NIST compliance controls. The JSON should look like a real policy excerpt an
organization might have — formal language, specific requirements, measurable
standards.

Output format: a JSON object with these keys:
- "document_type": always "policy"
- "title": a concise policy title
- "policy_text": the full policy text (3-6 sentences, formal tone)
- "department": plausible department name
- "classification": one of "public", "internal", "confidential"

Generate JSON that is AMBIGUOUS enough to plausibly map to MULTIPLE adjacent
controls, not just one. Real compliance documents never map cleanly to a
single control — they always touch multiple.

Return ONLY valid JSON, no markdown fences, no commentary."""

POLICY_USER = """Generate a {doc_type} document that maps to these NIST controls:

{controls}

The document should plausibly reference requirements that align with these
controls. Make the language realistic — this should read like something from
a real compliance department.

Generate JSON with {num_mappings} mappings. Include reasoning for each mapping.

Return ONLY the JSON."""


# ── Audit finding ───────────────────────────────────────────────────

FINDING_SYSTEM = """You are a security auditor writing realistic audit findings against NIST frameworks.

Your task: create a plausible audit finding JSON that an auditor would write
after reviewing a system against NIST compliance controls. Each finding should
reference specific control gaps and include severity, evidence, and remediation.

Output format: a JSON object with these keys:
- "finding_id": unique ID like "F-2025-001"
- "title": concise finding title
- "severity": one of "critical", "high", "medium", "low"
- "description": what was observed (2-4 sentences)
- "control_gaps": array of specific control gaps
- "remediation": recommended fix (2-3 sentences)

Generate JSON that is AMBIGUOUS enough to plausibly map to MULTIPLE adjacent
controls. Real audit findings rarely isolate to a single control.

Return ONLY valid JSON, no markdown fences, no commentary."""

FINDING_USER = """Generate an audit {doc_type} that maps to these NIST controls:

{controls}

The finding should describe a realistic control gap scenario. Make it
plausible — something you could actually find in a real audit.

Generate JSON with {num_mappings} mappings. Include reasoning for each mapping.

Return ONLY the JSON."""


# ── Risk assessment ─────────────────────────────────────────────────

RISK_SYSTEM = """You are a risk analyst writing realistic risk assessment entries.

Your task: create a plausible risk assessment JSON entry that evaluates risk
against NIST controls. Include likelihood, impact, inherent/ residual risk
scores, and treatment recommendations.

Output format: a JSON object with these keys:
- "risk_id": unique ID
- "risk_statement": what could happen (1-2 sentences)
- "likelihood": "low", "medium", "high", or "very_high"
- "impact": "low", "medium", "high", or "very_high"
- "inherent_risk": calculated risk level
- "residual_risk": risk after controls
- "treatment": "accept", "mitigate", "transfer", or "avoid"
- "treatment_description": description of treatment approach

Generate JSON that touches MULTIPLE controls — real risks never map to one.

Return ONLY valid JSON, no markdown fences, no commentary."""

RISK_USER = """Generate a {doc_type} entry that maps to these NIST controls:

{controls}

The risk should reflect a realistic scenario a compliance team would assess.

Generate JSON with {num_mappings} mappings. Include reasoning for each mapping.

Return ONLY the JSON."""


# ── Implementation description ──────────────────────────────────────

IMPLEMENTATION_SYSTEM = """You are a system architect describing control implementations.

Your task: create a plausible JSON description of how a specific NIST control
is implemented in a real system. Include technical details, tools used,
configuration specifics, and evidence of implementation.

Output format: a JSON object with these keys:
- "implementation_id": unique ID
- "control_area": which area this implementation covers
- "description": how the control is implemented (3-5 sentences)
- "technologies": array of tools or technologies used
- "configuration_details": specific config settings or approaches
- "evidence": how compliance can be verified
- "status": "implemented", "partially_implemented", "planned", or "not_applicable"

Generate JSON that naturally touches MULTIPLE adjacent controls — real
implementations span multiple controls.

Return ONLY valid JSON, no markdown fences, no commentary."""

IMPLEMENTATION_USER = """Generate an {doc_type} description that maps to these NIST controls:

{controls}

Describe implementation for a realistic enterprise environment.

Generate JSON with {num_mappings} mappings. Include reasoning for each mapping.

Return ONLY the JSON."""


# ── Gap analysis ────────────────────────────────────────────────────

GAP_SYSTEM = """You are a compliance analyst writing gap analysis entries.

Your task: create a plausible gap analysis JSON entry that identifies gaps
between current state and NIST control requirements. Include gap description,
current state, target state, and recommended actions.

Output format: a JSON object with these keys:
- "gap_id": unique ID
- "title": concise gap title
- "current_state": what currently exists (2-3 sentences)
- "target_state": what's required by the control (2-3 sentences)
- "gap_description": the gap between current and target (1-2 sentences)
- "recommended_actions": array of action items
- "priority": "critical", "high", "medium", or "low"
- "estimated_effort": "hours", "days", "weeks", or "months"

Generate JSON that touches MULTIPLE controls — real gaps almost always
span multiple related controls.

Return ONLY valid JSON, no markdown fences, no commentary."""

GAP_USER = """Generate a {doc_type} entry that maps to these NIST controls:

{controls}

The gap should reflect a realistic compliance gap an organization might have.

Generate JSON with {num_mappings} mappings. Include reasoning for each mapping.

Return ONLY the JSON."""


# ── Mapping output schema ───────────────────────────────────────────

MAPPING_SCHEMA = """Each mapping should be an object:
{
  "control_id": "SC-28",
  "control_title": "Protection of Information at Rest",
  "framework": "nist_800_53",
  "granularity": "control",
  "confidence": 0.94,
  "reasoning": "The JSON describes an encryption requirement for stored data which maps to SC-28 covering protection of information at rest."
}

Return the mappings in a "mappings" array alongside the document JSON:
{
  "document": { ... the generated policy/finding/risk JSON ... },
  "mappings": [ ... array of mapping objects ... ]
}"""


# ── Template registry ───────────────────────────────────────────────

TEMPLATES: dict[DocType, GenerationTemplate] = {
    "policy": GenerationTemplate(
        system=POLICY_SYSTEM + "\n\n" + MAPPING_SCHEMA,
        user_prefix=POLICY_USER,
    ),
    "finding": GenerationTemplate(
        system=FINDING_SYSTEM + "\n\n" + MAPPING_SCHEMA,
        user_prefix=FINDING_USER,
    ),
    "risk": GenerationTemplate(
        system=RISK_SYSTEM + "\n\n" + MAPPING_SCHEMA,
        user_prefix=RISK_USER,
    ),
    "implementation": GenerationTemplate(
        system=IMPLEMENTATION_SYSTEM + "\n\n" + MAPPING_SCHEMA,
        user_prefix=IMPLEMENTATION_USER,
    ),
    "gap": GenerationTemplate(
        system=GAP_SYSTEM + "\n\n" + MAPPING_SCHEMA,
        user_prefix=GAP_USER,
    ),
}

DOC_TYPES: list[DocType] = ["policy", "finding", "risk", "implementation", "gap"]


def _build_control_context(controls: list[ControlEntry]) -> str:
    """Build a structured description of controls for the prompt."""
    parts: list[str] = []
    for c in controls:
        fw = c.framework.value if isinstance(c.framework, Framework) else str(c.framework)
        parts.append(f"- [{fw}] {c.id}: {c.title} — {c.description}")
    return "\n".join(parts)
