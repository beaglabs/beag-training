"""Synthetic data generator — produces JSON → NIST mapping examples via DeepSeek.

Uses the framework catalog to randomly sample controls, then calls the
frontier model (DeepSeek) to generate plausible JSON that maps to those
controls.  Each example is validated against the output schema.

Strategy (from Bridgewater paper):
  1. Generate diverse examples at scale using frontier models
  2. Train initial LoRA model on synthetic data
  3. Route contested examples to auditors (~2–5 %)
  4. Retrain with auditor-labeled data
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from frameworks.catalog import Catalog, ControlEntry, Framework, Granularity, load_catalog
from frontier.deepseek import DeepSeekClient
from model.output_schema import ControlMapping, TrainingExample

from data.templates import DOC_TYPES, TEMPLATES, DocType


@dataclass
class GeneratorConfig:
    """Controls the scale and diversity of generated data."""

    total_examples: int = 1000
    examples_per_doc_type: int = 200
    max_mappings_per_example: int = 4
    min_mappings_per_example: int = 2
    concurrency: int = 10
    frameworks: list[Framework] = field(default_factory=lambda: list(Framework))
    granularities: list[Granularity] = field(default_factory=lambda: [
        Granularity.CONTROL, Granularity.ENHANCEMENT,
    ])
    # Probability of injecting deliberate ambiguity (adjacent controls)
    ambiguity_prob: float = 0.3


class DataGenerator:
    """Generates synthetic training examples by calling a frontier model."""

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        catalog: Catalog | None = None,
        config: GeneratorConfig | None = None,
    ):
        self._client = client
        self.catalog = catalog or load_catalog()
        self.config = config or GeneratorConfig()

    @property
    def client(self) -> DeepSeekClient:
        if self._client is None:
            self._client = DeepSeekClient()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> list[TrainingExample]:
        """Generate the full dataset synchronously."""
        return asyncio.run(self.generate_all_async())

    async def generate_all_async(self) -> list[TrainingExample]:
        tasks = self._build_generation_tasks()
        semaphore = asyncio.Semaphore(self.config.concurrency)
        results: list[TrainingExample] = []
        total_cost = 0.0

        async def run_one(prompt: dict[str, Any], idx: int) -> TrainingExample | None:
            nonlocal total_cost
            async with semaphore:
                return await self._generate_one(prompt, idx)

        print(f"\nGenerating {len(tasks)} synthetic examples "
              f"(concurrency={self.config.concurrency})...")
        t0 = time.monotonic()
        completed = 0
        total = len(tasks)

        async def run_one_with_progress(prompt, idx):
            nonlocal completed
            result = await run_one(prompt, idx)
            completed += 1
            if completed % 25 == 0 or completed == total:
                elapsed = time.monotonic() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed, ~{rate:.1f}/s, ETA {eta:.0f}s")
            return result

        outputs = await asyncio.gather(*(run_one_with_progress(p, i) for i, p in enumerate(tasks)))
        for ex in outputs:
            if ex is not None:
                results.append(ex)

        elapsed = time.monotonic() - t0
        print(f"  Generated {len(results)} valid examples in {elapsed:.1f}s")
        print(f"  Estimated cost: ${total_cost:.2f}")
        return results

    def save(
        self,
        examples: list[TrainingExample],
        path: Path | str,
        fmt: str = "jsonl",
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jsonl":
            with open(target, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_jsonl_dict(), ensure_ascii=False) + "\n")
        else:
            with open(target, "w") as f:
                json.dump(
                    [ex.to_jsonl_dict() for ex in examples],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        print(f"  Saved {len(examples)} examples to {target}")
        return target

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_generation_tasks(self) -> list[dict[str, Any]]:
        """Create a list of prompt configurations, one per example to generate."""
        tasks: list[dict[str, Any]] = []
        for i in range(self.config.total_examples):
            doc_type: DocType = DOC_TYPES[i % len(DOC_TYPES)]
            n = random.randint(
                self.config.min_mappings_per_example,
                self.config.max_mappings_per_example,
            )
            framework = random.choice(self.config.frameworks)
            controls = self._sample_controls(framework, n)
            template = TEMPLATES[doc_type]
            messages = template.build_messages(
                controls=controls,
                doc_type=doc_type,
                num_mappings=n,
            )
            tasks.append({
                "doc_type": doc_type,
                "messages": messages,
                "controls": [c.id for c in controls],
                "framework": framework.value if isinstance(framework, Framework) else str(framework),
            })
        return tasks

    def _sample_controls(
        self,
        framework: Framework,
        count: int,
    ) -> list[ControlEntry]:
        """Sample controls with optional injection of adjacent ambiguity."""
        entries = self.catalog.list_by_framework(framework)
        entries = [e for e in entries if e.granularity in self.config.granularities]
        if not entries:
            entries = self.catalog.list_by_framework(framework)

        if random.random() < self.config.ambiguity_prob and len(entries) > 2:
            # Pick a base, then choose adjacent controls
            base = random.choice(entries)
            selected = [base]
            others = [
                e for e in entries
                if e.parent_id == base.parent_id and e.id != base.id
            ]
            if others:
                selected.append(random.choice(others))
            if len(others) > 1:
                selected.append(random.choice(others))
            return selected[:count]

        return random.choices(entries, k=min(count, len(entries)))

    async def _generate_one(
        self,
        prompt: dict[str, Any],
        idx: int,
    ) -> TrainingExample | None:
        import openai

        messages = prompt["messages"]
        doc_type = prompt["doc_type"]

        try:
            response = await asyncio.wait_for(
                self.client.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=4096,
                ),
                timeout=60,
            )
        except openai.RateLimitError:
            await asyncio.sleep(2)
            return None
        except Exception:
            return None

        content = response.choices[0].message.content
        if not content:
            return None

        parsed = _parse_json_response(content)
        if parsed is None:
            return None

        doc = parsed.get("document", parsed)
        mappings_raw = parsed.get("mappings", [])

        mappings: list[ControlMapping] = []
        for m in mappings_raw:
            if not isinstance(m, dict):
                continue
            mappings.append(
                ControlMapping(
                    control_id=str(m.get("control_id", "")),
                    control_title=str(m.get("control_title", "")),
                    framework=str(m.get("framework", "")),
                    granularity=str(m.get("granularity", "")),
                    confidence=float(m.get("confidence", 0.9)),
                    reasoning=str(m.get("reasoning", "")),
                )
            )

        if not mappings:
            return None

        return TrainingExample(
            input_json=doc if isinstance(doc, dict) else {"raw": str(doc)},
            mappings=mappings,
            text_context=doc_type,
        )


def _parse_json_response(content: str) -> dict[str, Any] | None:
    """Robust JSON extraction from frontier model output."""
    content = content.strip()
    # Strip markdown fences
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    return None
