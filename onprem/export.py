"""ONNX export for Unsloth-trained models.

Workflow:
1. Merge LoRA into base model (unsloth merge_and_unload)
2. Save as HuggingFace model
3. Convert to ONNX via optimum-cli
4. Validate exported model against PyTorch
5. Save ONNX artifact to output directory
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def merge_and_export(
    adapter,
    output_dir: str,
    validate: bool = True,
) -> dict:
    """Full pipeline: merge LoRA, export ONNX, validate.

    Args:
        adapter: UnslothAdapter instance with loaded model
        output_dir: Directory for ONNX output
        validate: Run logit comparison validation

    Returns:
        Summary dict with export path and validation results
    """
    merged_model, tokenizer = adapter.merge_and_unload()

    hf_dir = Path(output_dir) / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    onnx_dir = Path(output_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    _run_optimum_export(str(hf_dir), str(onnx_dir))

    result = {
        "output_dir": output_dir,
        "onnx_dir": str(onnx_dir),
        "format": "ONNX",
        "validated": False,
        "cosine_similarity": None,
    }

    if validate:
        sim = validate_export(str(hf_dir), str(onnx_dir / "model.onnx"))
        result["validated"] = sim > 0.99
        result["cosine_similarity"] = round(sim, 6)

    return result


def export_only_lora(adapter, output_dir: str, validate: bool = True) -> dict:
    """Export LoRA adapter only (for cloud transfer). Then merge+export ONNX."""
    merged_model, tokenizer = adapter.merge_and_unload()

    hf_dir = Path(output_dir) / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    onnx_dir = Path(output_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    _run_optimum_export(str(hf_dir), str(onnx_dir))

    result = {
        "output_dir": output_dir,
        "onnx_dir": str(onnx_dir),
        "format": "ONNX",
    }

    if validate:
        sim = validate_export(str(hf_dir), str(onnx_dir / "model.onnx"))
        result["validated"] = sim > 0.99
        result["cosine_similarity"] = round(sim, 6)

    return result


def validate_export(
    hf_path: str,
    onnx_path: str,
    test_text: str = "The model should produce consistent outputs.",
) -> float:
    """Compare logits between original PyTorch and ONNX model.

    Must match within 99% cosine similarity.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        pt_model = AutoModelForCausalLM.from_pretrained(hf_path)

        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            pt_output = pt_model(**inputs)
            pt_logits = pt_output.logits.numpy()

        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {
            ort_session.get_inputs()[0].name: inputs["input_ids"].numpy(),
        }
        ort_outputs = ort_session.run(None, ort_inputs)[0]

        pt_flat = pt_logits.flatten().astype(np.float64)
        ort_flat = ort_outputs.flatten().astype(np.float64)

        similarity = _cosine_similarity(pt_flat, ort_flat)
        return similarity

    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def _run_optimum_export(model_path: str, output_dir: str) -> None:
    result = subprocess.run(
        [
            "optimum-cli", "export", "onnx",
            "--model", model_path,
            "--task", "text-generation",
            output_dir,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ONNX export failed:\n{result.stderr}")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
