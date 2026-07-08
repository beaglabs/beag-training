"""Unsloth adapter for on-premise training.

Loads base models, applies QLoRA with configurable rank,
supports merge/unload for inference and export.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None


class UnslothAdapter:
    """Wraps Unsloth's FastLanguageModel for Beag on-prem training.

    Supports the three model tiers from the cloud pipeline:
    - starter: Qwen3.5-4B
    - standard: Qwen2.5-7B-Instruct
    - performance: Qwen3.5-35B-A3B (Mixture of Experts)
    """

    MODEL_MAP = {
        "starter": "Qwen/Qwen2.5-3B-Instruct",
        "standard": "Qwen/Qwen2.5-7B-Instruct",
        "performance": "Qwen/Qwen2.5-32B-Instruct",
    }

    def __init__(
        self,
        model_tier: str = "standard",
        max_seq_length: int = 32768,
        load_in_4bit: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        device_map = None,
    ):
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Install with: "
                "pip install unsloth"
            )

        self.model_tier = model_tier
        self.model_id = self.MODEL_MAP.get(model_tier, self.MODEL_MAP["standard"])
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device_map = device_map

        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple:
        """Load base model and tokenizer with QLoRA applied."""
        load_kwargs = {
            "model_name": self.model_id,
            "max_seq_length": self.max_seq_length,
            "dtype": None,
            "load_in_4bit": self.load_in_4bit,
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )

        self.model.train()
        return self.model, self.tokenizer

    def load_for_inference(self, checkpoint_dir: str) -> Tuple:
        """Load a fine-tuned checkpoint for inference (no LoRA apply)."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self.model)
        return self.model, self.tokenizer

    def get_lora_weights(self) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights for storage."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora" in name:
                lora_params[name] = param.detach().clone()
        return lora_params

    def merge_and_unload(self) -> Tuple:
        """Merge LoRA into base and unload to single model. Returns (model, tokenizer)."""
        self.model = self.model.merge_and_unload()
        gc.collect()
        torch.cuda.empty_cache()
        return self.model, self.tokenizer

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def save_adapter(self, output_dir: str) -> None:
        """Save only the LoRA adapter weights (not the full model)."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
