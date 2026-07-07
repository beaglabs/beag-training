"""On-premise training entrypoint.

Runs the full local pipeline:
1. Load model via Unsloth (QLoRA)
2. Train with CISPO loss + interleaved batching
3. OPD recovery (optional)
4. Export to ONNX (optional)
5. Validate against PyTorch (optional)

Usage:
    python run.py --config config.yaml
    python run.py --config config.yaml --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from onprem.unsloth_adapter import UnslothAdapter
from onprem.train import OnPremTrainer, load_dataset_from_jsonl
from onprem.recipe import OPDRecovery, load_prompts_from_file
from onprem.export import merge_and_export


def main():
    parser = argparse.ArgumentParser(description="Beag On-Prem Model Training")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--validate", action="store_true", help="Run full validation pipeline")
    args = parser.parse_args()

    print("Loading config...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    customer = config.get("customer", {})
    training_config = config.get("training", {})
    opd_config = config.get("opd", {})
    export_config = config.get("export", {})
    checkpoint_config = config.get("checkpoints", {})

    print(f"Customer: {customer.get('id')}")
    print(f"Tier: {customer.get('model_tier', 'standard')}")

    # ─── Step 1: Load model ──────────────────────────────────────────
    print("\n[1/4] Loading model with Unsloth...")
    adapter = UnslothAdapter(
        model_tier=customer.get("model_tier", "standard"),
        max_seq_length=training_config.get("max_seq_length", 32768),
        load_in_4bit=True,
        lora_rank=training_config.get("lora_rank", 32),
        lora_alpha=training_config.get("lora_alpha", 16),
    )
    model, tokenizer = adapter.load()
    print(f"  Model: {adapter.model_id}")
    print(f"  LoRA rank: {training_config.get('lora_rank', 32)}")
    print(f"  Max seq length: {training_config.get('max_seq_length', 32768)}")

    # ─── Step 2: Train ───────────────────────────────────────────────
    print("\n[2/4] Training...")
    records = load_dataset_from_jsonl(training_config.get("dataset_path", "/data/training.jsonl"))
    print(f"  Loaded {len(records)} examples")

    trainer = OnPremTrainer(model, tokenizer, config=training_config)
    train_result = trainer.train(records)

    print(f"  Steps: {train_result['total_steps']}")
    print(f"  Final loss: {train_result['final_loss']:.4f}")
    print(f"  Best loss: {train_result['best_loss']:.4f}")

    ckpt_dir = Path(checkpoint_config.get("output_dir", "/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter.save(str(ckpt_dir / "trained"))

    # ─── Step 3: OPD Recovery ────────────────────────────────────────
    if opd_config.get("enabled", True):
        print("\n[3/4] OPD Recovery...")

        prompts_path = opd_config.get("prompts_path")
        if prompts_path and Path(prompts_path).exists():
            prompts = load_prompts_from_file(prompts_path)
            print(f"  Loaded {len(prompts)} prompts")

            teacher_adapter = UnslothAdapter(
                model_tier=customer.get("model_tier", "standard"),
                max_seq_length=training_config.get("max_seq_length", 32768),
                load_in_4bit=True,
            )
            teacher_model, _ = teacher_adapter.load()
            teacher_model.eval()

            opd = OPDRecovery(model, teacher_model, tokenizer, opd_config)
            opd_result = opd.recover(prompts)

            print(f"  OPD steps: {opd_result['opd_steps']}")
            print(f"  Final KL: {opd_result['final_loss']:.4f}")
            print(f"  Best KL: {opd_result['best_loss']:.4f}")

            adapter.save(str(ckpt_dir / "recovered"))

            teacher_adapter.unload()
            del teacher_adapter
        else:
            print("  Skipped: no prompts file found")

    # ─── Step 4: Export ONNX ─────────────────────────────────────────
    if export_config.get("format") == "onnx":
        print("\n[4/4] Exporting ONNX...")

        export_dir = Path(export_config.get("output_dir", "/exports"))
        export_dir.mkdir(parents=True, exist_ok=True)

        validate = export_config.get("validate", args.validate)
        result = merge_and_export(
            adapter,
            output_dir=str(export_dir),
            validate=validate,
        )

        print(f"  ONNX dir: {result['onnx_dir']}")
        if result.get("validated"):
            print(f"  Validation: PASSED (cosine sim: {result['cosine_similarity']})")
        else:
            print(f"  Validation: SKIPPED")

    # ─── Summary ─────────────────────────────────────────────────────
    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "customer_id": customer.get("id"),
        "model_tier": customer.get("model_tier"),
        "training": train_result,
        "export_path": str(export_config.get("output_dir", "/exports")),
    }

    summary_path = Path(export_config.get("output_dir", "/exports")) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nDone! Summary: {summary_path}")
    adapter.unload()


if __name__ == "__main__":
    main()
