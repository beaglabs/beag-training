# Beag Model Training

Train domain-specific LoRA adapters that map documents to NIST compliance controls. Based on the [Bridgewater / Thinking Machines recipe](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/): synthetic data generation, QLoRA fine-tuning with CISPO loss, and on-policy distillation.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,dev]"
cp .env.example .env   # then edit .env with your DEEPSEEK_API_KEY
```

## Kaggle Notebook (T4 GPU)

Open `notebooks/kaggle_train.ipynb` in Kaggle — it clones this repo, generates data, trains a Qwen2.5-7B QLoRA adapter, and saves the weights. Total runtime ~15-25 min.

## Generate synthetic NIST data

Requires `DEEPSEEK_API_KEY` in your environment.

```bash
# 500 examples (default)
python train.py --generate --nist 500 --output ./output

# Generate standalone (no training), with higher diversity
python scripts/generate_more.py
```

Edit `scripts/generate_more.py` to tune the number of examples, concurrent requests, ambiguity probability, and mappings per example.

## Train

```bash
# Full pipeline: generate → train → ONNX export
python train.py --generate --nist 500 --task nist --no-label --tier standard --export

# Train on existing labeled data only
python train.py --data output/generated_augmented.jsonl --task nist --no-label --export

# On-prem training from config
python onprem/run.py --config onprem/config.yaml
```

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | — | CSV, Parquet, or JSONL input file |
| `--generate` | off | Generate synthetic NIST data first |
| `--nist` | 500 | Number of synthetic examples to generate |
| `--task` | `classification` | One of: `classification`, `extraction`, `code`, `custom`, `nist` |
| `--tier` | `standard` | Model tier: `starter`, `standard`, `performance` |
| `--no-label` | off | Skip DeepSeek labeling (use existing labels) |
| `--no-opd` | off | Skip on-policy distillation recovery |
| `--no-export` | off | Skip ONNX export |
| `--batch-size` | 4 | Training batch size |
| `--epochs` | 3 | Number of train epochs |
| `--lora-rank` | 32 | LoRA rank |
| `--max-seq-length` | 32768 | Max sequence length |

## Auditor active learning (Phase 3)

The pipeline follows the Bridgewater paper's 4-step strategy:

1. **Generate** synthetic data via frontier models (DeepSeek)
2. **Train** initial LoRA adapter
3. **Route contested examples** — run inference, score disagreement via KL divergence, send top ~2–5% to a human auditor
4. **Retrain** with auditor-corrected labels

Contested-detection logic lives in `disagreement/scorer.py`. The auditor review interface and automated retrain loop are planned but not yet built.

## Project layout

```
beag-training/
├── train.py              # Single-command CLI (generate → train → export)
├── scripts/
│   └── generate_more.py   # Standalone synthetic data generation
├── core/
│   └── config.py         # Settings (reads DEEPSEEK_API_KEY from env)
├── data/
│   ├── generator.py      # Synthetic example generation via DeepSeek
│   ├── augment.py        # Data augmentation (paraphrase, negate, swap)
│   ├── validator.py      # Validate mappings against framework catalog
│   └── templates/        # Prompt templates per doc type
├── disagreement/
│   └── scorer.py         # KL-divergence contested example selection
├── frameworks/
│   ├── catalog.py        # NIST 800-53, CSF, CMMC control catalog
│   └── fetchers/         # Control data fetchers per framework
├── frontier/
│   ├── deepseek.py       # DeepSeek API client (cold-start labeling)
│   └── prompts.py        # Labeling prompt construction
├── model/
│   └── output_schema.py  # TrainingExample, ControlMapping schemas
├── onprem/
│   ├── run.py            # On-prem training entrypoint
│   ├── train.py          # Training loop (CISPO loss, interleaved batching)
│   ├── recipe.py         # OPD recovery phase
│   ├── export.py         # ONNX export with validation
│   ├── config.yaml       # On-prem training config
│   └── Dockerfile        # Dockerized training
└── output/               # Generated datasets and checkpoints
```

## Environment

Copy `.env.example` to `.env` and fill in:

```bash
DEEPSEEK_API_KEY=         # Required for synthetic data generation
TINKER_API_KEY=           # Optional: Tinker cloud training
```
