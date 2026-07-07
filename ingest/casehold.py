"""CaseHOLD dataset loader for legal citation classification.

Downloads the CaseHOLD dataset (legal text → correct citation ruling),
splits into train/validation/test, and exports as CSV for upload to the
training pipeline.

CaseHOLD: https://huggingface.co/datasets/lexlms/casehold
Task: Given a legal text excerpt and 5 possible holdings (legal rulings),
      select the correct one that applies to the cited case.

Run:
    python -m ingest.casehold --output-dir ./data/casehold
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


DATASET_ID = "lexlms/casehold"
HOLDING_LABELS = ["holding_0", "holding_1", "holding_2", "holding_3", "holding_4"]


def download_casehold(output_dir: Path, sample_size: Optional[int] = None) -> None:
    """Download CaseHOLD from HuggingFace and export as CSV."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)

    train = dataset["train"]
    val = dataset.get("validation")
    test = dataset.get("test")

    if val is None:
        split = train.train_test_split(test_size=0.1, seed=42)
        train, val = split["train"], split["test"]

    if test is None and val is not None:
        split = val.train_test_split(test_size=0.5, seed=42)
        val, test = split["train"], split["test"]

    if sample_size:
        train = train.select(range(min(sample_size, len(train))))
        val = val.select(range(min(sample_size // 5, len(val)))) if val else None
        test = test.select(range(min(sample_size // 5, len(test)))) if test else None

    def _export(split_data, filename, label):
        if split_data is None:
            return
        rows = []
        for example in split_data:
            rows.append({
                "text": example["context"],
                "holding_0": example.get("holdings", [""] * 5)[0] if "holdings" in example else "",
                "holding_1": example.get("holdings", [""] * 5)[1] if "holdings" in example else "",
                "holding_2": example.get("holdings", [""] * 5)[2] if "holdings" in example else "",
                "holding_3": example.get("holdings", [""] * 5)[3] if "holdings" in example else "",
                "holding_4": example.get("holdings", [""] * 5)[4] if "holdings" in example else "",
                "correct_holding": str(example.get("label", example.get("holding_id", 0))),
            })

        path = output_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {filename}: {len(rows)} examples -> {path}")

    _export(train, "casehold_train.csv", "train")
    _export(val, "casehold_val.csv", "validation")
    if test:
        _export(test, "casehold_test.csv", "test")

    print(f"\nDone. Upload casehold_train.csv to /model-service/runs/create.")
    print("Use 'correct_holding' as the label column.")


def main():
    parser = argparse.ArgumentParser(description="Download CaseHOLD dataset")
    parser.add_argument("--output-dir", default="./data/casehold", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Limit examples (for quick testing)")
    args = parser.parse_args()

    download_casehold(Path(args.output_dir), sample_size=args.sample)


if __name__ == "__main__":
    main()
