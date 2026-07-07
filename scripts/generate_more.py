#!/usr/bin/env python3
"""Generate additional synthetic NIST data and merge with existing."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("Error: DEEPSEEK_API_KEY not set in environment", file=sys.stderr)
    sys.exit(1)

from data import DataGenerator, GeneratorConfig
from data.validator import filter_valid
from data.augment import DataAugmenter
from frameworks.catalog import Framework, load_catalog
from model.output_schema import TrainingExample


def load_existing_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_existing_examples(path: Path) -> list[TrainingExample]:
    examples = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if "input_json" in d:
                        ex = TrainingExample.from_jsonl_dict(d)
                        examples.append(ex)
    return examples


async def main():
    output_dir = Path("output")
    target_new = 800
    cat = load_catalog()

    print("=" * 60)
    print(f"Generating {target_new} new synthetic NIST compliance examples...")
    print("=" * 60)

    config = GeneratorConfig(
        total_examples=target_new,
        frameworks=list(Framework),
        concurrency=10,
        max_mappings_per_example=6,
        min_mappings_per_example=3,
        ambiguity_prob=0.5,
    )
    gen = DataGenerator(catalog=cat, config=config)
    t0 = time.monotonic()
    import sys
    sys.stdout.flush()
    new_examples = await gen.generate_all_async()
    sys.stdout.flush()
    elapsed = time.monotonic() - t0

    new_valid, new_invalid = filter_valid(new_examples, cat)
    print(f"  Generated {len(new_examples)} raw → {len(new_valid)} valid, {len(new_invalid)} invalid")
    print(f"  Time: {elapsed:.1f}s")

    # Augment ~20% of new examples
    augmenter = DataAugmenter(cat)
    augmented: list = []
    augment_count = max(1, len(new_valid) // 5)
    for ex in new_valid[:augment_count]:
        augmented.extend(augmenter.augment(ex))
    aug_valid, aug_invalid = filter_valid(augmented, cat)
    print(f"  Augmented: {len(aug_valid)} valid, {len(aug_invalid)} invalid")

    # Save new batch
    all_new = new_valid + aug_valid
    new_jsonl_path = output_dir / "generated_new.jsonl"
    with open(new_jsonl_path, "w") as f:
        for ex in all_new:
            record = {
                "text": json.dumps(ex.input_json),
                "mappings": [m.to_dict() for m in ex.mappings],
                "text_context": ex.text_context,
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved new batch: {new_jsonl_path} ({len(all_new)} records)")

    # Merge with existing generated_augmented.jsonl
    existing_path = output_dir / "generated_augmented.jsonl"
    existing_records = load_existing_jsonl(existing_path)
    print(f"\n  Existing records: {len(existing_records)}")

    # Back up existing
    backup_path = output_dir / "generated_augmented.bak.jsonl"
    with open(backup_path, "w") as f:
        for rec in existing_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Backed up to: {backup_path}")

    # Load new records as plain dicts (from the saved file)
    new_records = load_existing_jsonl(new_jsonl_path)

    # Add unique source marker to new records
    for rec in new_records:
        rec["source_batch"] = "batch_2026-07-07_800"

    merged = existing_records + new_records
    with open(existing_path, "w") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")

    # Also update generated.jsonl with just the raw generated (not augmented)
    gen_path = output_dir / "generated.jsonl"
    existing_gen = load_existing_jsonl(gen_path)
    # For generated.jsonl we save TrainingExample format
    new_gen_records = []
    for ex in new_valid:
        record = {
            "text": json.dumps(ex.input_json),
            "mappings": [m.to_dict() for m in ex.mappings],
            "text_context": ex.text_context,
        }
        new_gen_records.append(record)
    merged_gen = existing_gen + new_gen_records
    with open(gen_path, "w") as f:
        for rec in merged_gen:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  New valid generated:  {len(new_valid)}")
    print(f"  New augmented:        {len(aug_valid)}")
    print(f"  Total new added:      {len(all_new)}")
    print(f"  Existing records:     {len(existing_records)}")
    print(f"  Merged total:         {len(merged)}")
    print(f"  Output:               {existing_path}")
    print(f"  Backup:               {backup_path}")


if __name__ == "__main__":
    asyncio.run(main())
