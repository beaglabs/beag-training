"""DeepSeek API client for cold start labeling.

Uses the prompt library to construct per-task-type prompts with
optional few-shot examples and custom label sets.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from openai import AsyncOpenAI

from core.config import settings
from frontier.prompts import build_messages, TaskType


@dataclass
class TaskDefinition:
    """Defines the labeling task for a training run."""

    type: TaskType = "classification"
    labels: list[str] = field(default_factory=lambda: ["category_a", "category_b", "category_c"])
    query: str = ""
    fields: list[str] = field(default_factory=list)
    description: str = ""
    language: str = "unknown"
    few_shot_examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "labels": self.labels,
            "query": self.query,
            "fields": self.fields,
            "description": self.description,
            "language": self.language,
            "few_shot_count": len(self.few_shot_examples),
        }


@dataclass
class LabelingResult:
    label: str
    tokens_used: int
    cost: float
    latency_ms: float


class DeepSeekClient:
    """Client for DeepSeek API used for cold start labeling."""

    COST_PER_1M_INPUT = 0.14   # $0.14 per 1M input tokens
    COST_PER_1M_OUTPUT = 0.28  # $0.28 per 1M output tokens

    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key or settings.deepseek_api_key,
            timeout=60.0,
            max_retries=2,
        )

    async def label_example(
        self,
        text: str,
        task: TaskDefinition | None = None,
        system_prompt: str | None = None,
        label_options: list[str] | None = None,
    ) -> LabelingResult:
        start = time.monotonic()

        if task is not None:
            messages = build_messages(
                task_type=task.type,
                text=text,
                task_description=task.description,
                labels=task.labels,
                query=task.query,
                fields=task.fields,
                language=task.language,
                few_shot_examples=task.few_shot_examples if task.few_shot_examples else None,
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": text},
            ]

        response = await self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )

        elapsed = (time.monotonic() - start) * 1000
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = (usage.total_tokens if usage else 0)

        cost = (
            (input_tokens / 1_000_000) * self.COST_PER_1M_INPUT
            + (output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT
        )

        label = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

        return LabelingResult(
            label=label,
            tokens_used=total_tokens,
            cost=cost,
            latency_ms=elapsed,
        )

    async def batch_label(
        self,
        examples: list[dict],
        task: TaskDefinition | None = None,
        system_prompt: str | None = None,
        label_options: list[str] | None = None,
        concurrency: int = 10,
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(concurrency)
        total_cost = 0.0
        total_tokens = 0

        async def label_one(ex: dict) -> dict:
            nonlocal total_cost, total_tokens
            async with semaphore:
                text = ex.get("input_data", {}).get("text", "")
                if not text:
                    code = ex.get("input_data", {}).get("code", "")
                    if code:
                        text = code

                result = await self.label_example(
                    text=text,
                    task=task,
                    system_prompt=system_prompt,
                    label_options=label_options,
                )
                total_cost += result.cost
                total_tokens += result.tokens_used
                return {
                    **ex,
                    "frontier_label": result.label,
                    "labeling_cost": result.cost,
                    "labeling_tokens": result.tokens_used,
                }

        results = await asyncio.gather(*(label_one(ex) for ex in examples))
        return results

    async def estimate_cost(
        self,
        example_count: int,
        avg_input_chars: int = 500,
        few_shot_count: int = 0,
    ) -> dict:
        est_input_tokens = example_count * (avg_input_chars // 3 + few_shot_count * (avg_input_chars // 2 + 5))
        est_output_tokens = example_count * 3
        est_cost = (
            (est_input_tokens / 1_000_000) * self.COST_PER_1M_INPUT
            + (est_output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT
        )
        return {
            "estimated_tokens": est_input_tokens + est_output_tokens,
            "estimated_cost": round(est_cost, 4),
            "model": settings.deepseek_model,
            "few_shot_count": few_shot_count,
        }
