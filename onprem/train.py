"""On-premise training loop with CISPO loss and interleaved batching.

Feature-parity with the Tinker cloud pipeline:
- CISPO asymmetric loss (Bridgewater paper)
- Interleaved batching for multi-task datasets
- Cosine warmup schedule
- Gradient accumulation
- Checkpoint saving
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class LabeledDataset(Dataset):
    """Dataset for on-prem training. Reads from JSONL or in-memory list."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        text_field: str = "text",
        label_field: str = "corrected_label",
        max_seq_length: int = 32768,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = record.get(self.text_field, "")
        if isinstance(text, dict):
            text = text.get("text", str(text))

        label = record.get(self.label_field, record.get("frontier_label", ""))
        task_id = record.get("task_id", None)

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": label,
            "task_id": task_id,
        }


class CISPOLoss(torch.nn.Module):
    """CISPO loss with asymmetric clipping.

    Based on the Bridgewater/Thinking Machines paper (arXiv:2510.13786).

    The asymmetric clip ranges prevent:
    - Over-confidence (high end is aggressive)
    - Under-confidence (low end is lenient)

    Args:
        clip_low: Lower clip threshold (default 0.2)
        clip_high: Upper clip threshold (default 0.8)
        beta: Weight for the entropy bonus (default 0.1)
    """

    def __init__(self, clip_low: float = 0.2, clip_high: float = 0.8, beta: float = 0.1):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        clipped_probs = torch.clamp(target_probs, self.clip_low, self.clip_high)
        nll = -torch.log(clipped_probs)

        mask = (target_probs > self.clip_low).float()

        entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)
        entropy_bonus = self.beta * entropy

        loss = nll * mask - entropy_bonus

        if attention_mask is not None:
            loss = loss * attention_mask
            norm = attention_mask.sum().clamp(min=1)
            loss = loss.sum() / norm

        return loss.mean()


def interleaved_collate(batch: list[dict]) -> dict:
    """Interleave examples from different tasks for multi-task training.

    Groups examples by task_id, then round-robins through groups.
    This ensures each batch contains examples from all tasks.
    """
    task_groups: dict = defaultdict(list)
    for item in batch:
        tid = item.get("task_id", "__default__")
        task_groups[tid].append(item)

    interleaved = []
    max_group_size = max(len(g) for g in task_groups.values())

    for i in range(max_group_size):
        for tid in task_groups:
            if i < len(task_groups[tid]):
                interleaved.append(task_groups[tid][i])

    input_ids = torch.stack([item["input_ids"] for item in interleaved])
    attention_mask = torch.stack([item["attention_mask"] for item in interleaved])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": [item["label"] for item in interleaved],
    }


class OnPremTrainer:
    """Self-contained training loop for on-premise deployment."""

    def __init__(
        self,
        model,
        tokenizer,
        config: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.loss_fn = CISPOLoss(
            clip_low=config.get("cispo_clip_low", 0.2),
            clip_high=config.get("cispo_clip_high", 0.8),
            beta=config.get("cispo_beta", 0.1),
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        self.use_interleaved = config.get("interleaved_batching", True)
        self.batch_size = config.get("batch_size", 4)
        self.gradient_accumulation_steps = config.get("gradient_accumulation", 4)
        self.num_epochs = config.get("num_epochs", 3)
        self.max_steps = config.get("max_steps", 500)
        self.warmup_steps = config.get("warmup_steps", 50)
        self.save_every = config.get("save_every", 100)
        self.output_dir = config.get("output_dir", "output")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        self.global_step = 0
        self.best_loss = float("inf")

    def train(self, records: list[dict]) -> dict:
        """Run the full training loop. Returns training summary."""
        dataset = LabeledDataset(
            records,
            self.tokenizer,
            text_field=self.config.get("text_field", "text"),
            label_field=self.config.get("label_field", "corrected_label"),
            max_seq_length=self.config.get("max_seq_length", 32768),
        )

        collate_fn = interleaved_collate if self.use_interleaved else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_steps = min(
            self.num_epochs * len(loader) // self.gradient_accumulation_steps,
            self.max_steps,
        )

        history = {"loss": [], "lr": [], "step": []}

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            print(f"Epoch {epoch + 1}/{self.num_epochs} ({len(loader)} batches)...", flush=True)

            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

                labels = input_ids[:, 1:]
                logits = logits[:, :-1, :]
                mask = attention_mask[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), mask.reshape(-1))
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self._step_scheduler(total_steps)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item()

                history["loss"].append(loss.item() * self.gradient_accumulation_steps)
                history["lr"].append(self.optimizer.param_groups[0]["lr"])
                history["step"].append(self.global_step)

                if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                    print(f"  batch {batch_idx}/{len(loader)} | loss {loss.item()*self.gradient_accumulation_steps:.4f} | step {self.global_step}/{total_steps}", flush=True)

                if self.global_step % self.save_every == 0 and self.global_step > 0:
                    self._save_checkpoint()

                if self.global_step >= total_steps:
                    break

            avg_loss = epoch_loss / max(len(loader), 1)
            print(f"  epoch {epoch + 1} avg loss: {avg_loss:.4f}", flush=True)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint("best")

        self._save_checkpoint("final")

        return {
            "total_steps": self.global_step,
            "final_loss": history["loss"][-1] if history["loss"] else 0,
            "best_loss": self.best_loss,
            "history": history,
        }

    def _step_scheduler(self, total_steps: int) -> None:
        if self.global_step < self.warmup_steps:
            ratio = self.global_step / max(self.warmup_steps, 1)
        else:
            progress = (self.global_step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
            ratio = max(0, 0.5 * (1 + math.cos(math.pi * progress)))

        base_lr = self.config.get("learning_rate", 2e-5)
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * ratio

    def _save_checkpoint(self, name: str | None = None) -> None:
        step = self.global_step
        tag = name or f"step-{step}"
        path = Path(self.output_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        (path / "trainer_state.json").write_text(json.dumps({
            "step": step,
            "loss": self.best_loss,
        }))


def load_dataset_from_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
