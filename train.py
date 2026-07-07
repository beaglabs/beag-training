#!/usr/bin/env python3
"""
Beag Model Training — single-command training recipe.

Workflow:
  0. Generate synthetic data via DeepSeek (optional, --generate)
  1. Ingest CSV/Parquet/JSONL → labeled examples (optional DeepSeek cold-start)
  2. Train with Unsloth QLoRA + CISPO loss
  3. OPD recovery (optional)
  4. ONNX export with validation

Usage:
  # Generate synthetic NIST compliance data
  python train.py --generate --nist 1000 --output ./output

  # Train on existing data
  python train.py --data input.csv --labels "bug,clean,anti-pattern" --tier standard
  python train.py --data input.parquet --task extraction --tier performance --no-label
  python train.py --data labeled.jsonl --no-label --export

  # Full pipeline: generate -> train -> export
  python train.py --generate --nist 500 --labels "" --tier standard --export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from ingest import detect_format, normalize, table_to_examples
from frontier.deepseek import DeepSeekClient, TaskDefinition
from onprem.unsloth_adapter import UnslothAdapter
from onprem.train import OnPremTrainer, load_dataset_from_jsonl
from onprem.recipe import OPDRecovery, load_prompts_from_file
from onprem.export import merge_and_export


def main():
    parser = argparse.ArgumentParser(description="Beag Model Training")
    parser.add_argument("--data", default="", help="CSV, Parquet, or JSONL file")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic NIST data before training")
    parser.add_argument("--nist", type=int, default=500, help="Number of synthetic NIST examples to generate")
    parser.add_argument("--task", default="classification", choices=["classification", "extraction", "code", "custom", "nist"])
    parser.add_argument("--labels", default="", help="Comma-separated label set (e.g. 'bug,clean')")
    parser.add_argument("--tier", default="standard", choices=["starter", "standard", "performance"])
    parser.add_argument("--description", default="", help="Task description for the labeler")
    parser.add_argument("--no-label", dest="skip_label", action="store_true", help="Skip DeepSeek labeling (use existing labels)")
    parser.add_argument("--no-opd", dest="skip_opd", action="store_true", help="Skip OPD recovery")
    parser.add_argument("--no-export", dest="skip_export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--prompts", default="", help="Path to OPD prompts JSONL file")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_list = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []
    task_type = args.task

    # ─── Step 0: Generate synthetic NIST data ─────────────────────────
    if args.generate:
        print(f"\n{'='*60}")
        print(f"[0/5] Generating synthetic NIST compliance data...")
        print(f"  Target examples: {args.nist}")

        from data import DataGenerator, GeneratorConfig
        from data.validator import filter_valid
        from frameworks.catalog import Framework, load_catalog

        cat = load_catalog()
        config = GeneratorConfig(
            total_examples=args.nist,
            frameworks=list(Framework),
            concurrency=10,
        )
        gen = DataGenerator(catalog=cat, config=config)
        examples = gen.generate_all()
        examples, invalid = filter_valid(examples, cat)
        print(f"  Valid: {len(examples)}  Invalid: {len(invalid)}")

        gen.save(examples, output_dir / "generated.jsonl")

        from data.augment import DataAugmenter

        augmenter = DataAugmenter(cat)
        augmented: list = []
        for ex in examples[: max(1, len(examples) // 5)]:
            augmented.extend(augmenter.augment(ex))
        aug_valid, aug_invalid = filter_valid(augmented, cat)
        print(f"  Augmented: {len(aug_valid)} valid, {len(aug_invalid)} invalid")

        all_examples = examples + aug_valid
        labeled_path = output_dir / "generated_augmented.jsonl"
        with open(labeled_path, "w") as f:
            for ex in all_examples:
                record = {
                    "text": json.dumps(ex.input_json),
                    "mappings": [m.to_dict() for m in ex.mappings],
                    "text_context": ex.text_context,
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved: {labeled_path} ({len(all_examples)} examples)")

        skip_to_train = True
    else:
        if not args.data:
            print("Error: --data is required when not using --generate")
            sys.exit(1)
        labeled_path = output_dir / "labeled.jsonl"
        skip_to_train = False

    # ─── Step 1: Ingest ────────────────────────────────────────────────
    if not skip_to_train:
        print(f"\n{'='*60}")
        print(f"[1/5] Loading data from {args.data}...")

        data_path = Path(args.data)
        fmt = detect_format(data_path.name)
        raw = data_path.read_bytes()
        table = normalize(raw, fmt)
        examples = table_to_examples(table)

        print(f"  Format: {fmt}")
        print(f"  Examples: {len(examples)}")
        print(f"  Columns: {table.column_names}")

        # ─── Step 2: DeepSeek Labeling ──────────────────────────────
        if not args.skip_label and label_list:
            print(f"\n{'='*60}")
            print(f"[2/5] Frontier labeling via DeepSeek...")
            print(f"  Task: {task_type}")
            print(f"  Labels: {label_list}")
            print(f"  Concurrency: 10")

            task_def = TaskDefinition(
                type=task_type,
                labels=label_list,
                description=args.description,
            )

            client = DeepSeekClient()

            async def run_label():
                return await client.batch_label(examples=examples, task=task_def)

            labeled = asyncio.run(run_label())
            examples = labeled

            total_cost = sum(e.get("labeling_cost", 0) for e in labeled)
            total_tokens = sum(e.get("labeling_tokens", 0) for e in labeled)
            print(f"  Done. Tokens: {total_tokens:,}  Cost: ${total_cost:.2f}")
        else:
            print(f"\n{'='*60}")
            print(f"[2/5] Skipping frontier labeling (--no-label or no labels specified)")

        with open(labeled_path, "w") as f:
            for ex in examples:
                record = {
                    "text": ex.get("input_data", {}).get("text", str(ex.get("input_data", {}))),
                    "frontier_label": ex.get("frontier_label", ex.get("original_label", "")),
                    "original_label": ex.get("original_label"),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved: {labeled_path}")
    else:
        print(f"\n{'='*60}")
        print(f"[1/5][2/5] Skipping ingest & labeling (generated data ready)")

    # ─── Step 3: Train ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[3/5] Training with Unsloth QLoRA...")
    print(f"  Tier: {args.tier}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Max sequence: {args.max_seq_length}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    adapter = UnslothAdapter(
        model_tier=args.tier,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
    )
    model, tokenizer = adapter.load()

    records = load_dataset_from_jsonl(str(labeled_path))
    if not records:
        print("  No training records found. Skipping training.")
    else:
        # Check if data has NIST-style mappings (multi-label)
        has_mappings = any("mappings" in r for r in records)

        if has_mappings and args.task == "nist":
            print(f"  Multi-label NIST training: {len(records)} examples")
            from frameworks.catalog import load_catalog
            nist_cat = load_catalog()
            trainer_config = {
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "output_dir": str(output_dir / "checkpoints"),
                "task_type": "nist_multi_label",
                "catalog": nist_cat,
            }
        else:
            trainer_config = {
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "output_dir": str(output_dir / "checkpoints"),
            }

        trainer = OnPremTrainer(model, tokenizer, config=trainer_config)
        train_result = trainer.train(records)

        print(f"  Steps: {train_result['total_steps']}")
        final_loss = train_result.get("final_loss")
        best_loss = train_result.get("best_loss")
        if final_loss is not None:
            print(f"  Final loss: {final_loss:.4f}")
        if best_loss is not None:
            print(f"  Best loss: {best_loss:.4f}")

        adapter.save(str(output_dir / "checkpoints" / "trained"))

    # ─── Step 4: OPD Recovery ──────────────────────────────────────────
    if not args.skip_opd and args.prompts and Path(args.prompts).exists():
        print(f"\n{'='*60}")
        print(f"[4/5] OPD Recovery...")

        prompts = load_prompts_from_file(args.prompts)
        print(f"  Prompts: {len(prompts)}")

        teacher_adapter = UnslothAdapter(
            model_tier=args.tier,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        teacher_model, _ = teacher_adapter.load()
        teacher_model.eval()

        opd = OPDRecovery(model, teacher_model, tokenizer, {"opd_steps": 50})
        opd_result = opd.recover(prompts)

        print(f"  OPD steps: {opd_result['opd_steps']}")
        print(f"  Final KL: {opd_result['final_loss']:.4f}")
        print(f"  Best KL: {opd_result['best_loss']:.4f}")

        adapter.save(str(output_dir / "checkpoints" / "recovered"))
        teacher_adapter.unload()
    else:
        print(f"\n{'='*60}")
        print(f"[4/5] Skipping OPD recovery (--no-opd or no prompts)")

    # ─── Step 5: Export ONNX ───────────────────────────────────────────
    if not args.skip_export:
        print(f"\n{'='*60}")
        print(f"[5/5] Exporting ONNX...")

        result = merge_and_export(adapter, output_dir=str(output_dir / "export"), validate=True)

        print(f"  ONNX dir: {result['onnx_dir']}")
        if result.get("validated"):
            print(f"  Validation: PASSED (cosine sim: {result['cosine_similarity']})")
        else:
            print(f"  Validation: SKIPPED")
    else:
        print(f"\n{'='*60}")
        print(f"[5/5] Skipping ONNX export (--no-export)")

    # ─── Summary ───────────────────────────────────────────────────────
    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "task": task_type,
        "labels": label_list,
        "tier": args.tier,
        "examples": len(examples),
        "output_dir": str(output_dir),
    }
    print(f"\nSummary saved to {output_dir / 'summary.json'}")
    print("\nDone.")

    adapter.unload()


if __name__ == "__main__":
    main()
