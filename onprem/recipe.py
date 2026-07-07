"""OPD (On-Policy Distillation) recovery phase for on-premise training.

After domain fine-tuning, the model can lose instruction-following behavior.
OPD recovers this by distilling from a teacher (Qwen3.5-9B-Instruct) over
generic instruction prompts (Tulu-3 dataset).

The OPD algorithm:
1. Student samples a response from each prompt
2. Teacher computes logprobs on student tokens
3. Minimize KL(student || teacher) — pushes student toward teacher distribution
4. Every N steps, promote best checkpoint as the new teacher target
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class OPDRecovery:
    """On-policy distillation to recover instruction-following behavior."""

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        config: dict,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.config = config

        self.steps = config.get("opd_steps", 50)
        self.batch_size = config.get("opd_batch_size", 2)
        self.max_new_tokens = config.get("opd_max_tokens", 512)
        self.learning_rate = config.get("opd_lr", 5e-6)
        self.promote_every = config.get("promote_every", 10)
        self.temperature = config.get("temperature", 0.7)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
        )

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.best_loss = float("inf")

    def recover(self, prompts: list[str]) -> dict:
        """Run OPD recovery over a set of prompts. Returns summary stats."""
        history = []

        for step in range(self.steps):
            batch = self._sample_batch(prompts)

            self.student.train()
            self.optimizer.zero_grad()

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_seq_length", 32768) - self.max_new_tokens,
            ).to(self.device)

            with torch.no_grad():
                student_output = self.student.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            student_logprobs = self._compute_logprobs(self.student, student_output)
            teacher_logprobs = self._compute_logprobs(self.teacher, student_output)

            loss = F.kl_div(
                F.log_softmax(student_logprobs / self.temperature, dim=-1),
                F.softmax(teacher_logprobs / self.temperature, dim=-1),
                reduction="batchmean",
            ) * (self.temperature ** 2)

            loss.backward()
            self.optimizer.step()

            history.append(loss.item())

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()

            if (step + 1) % self.promote_every == 0:
                self._promote_teacher()

        return {
            "opd_steps": self.steps,
            "initial_loss": history[0] if history else 0,
            "final_loss": history[-1] if history else 0,
            "best_loss": self.best_loss,
            "history": history,
        }

    def _compute_logprobs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(input_ids)
            return F.log_softmax(outputs.logits, dim=-1)

    def _sample_batch(self, prompts: list[str]) -> list[str]:
        indices = torch.randint(0, len(prompts), (self.batch_size,)).tolist()
        return [prompts[i] for i in indices]

    def _promote_teacher(self) -> None:
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()


def load_prompts_from_file(path: str) -> list[str]:
    """Load instruction prompts from a JSONL file (Tulu-3 format).

    Expected format:
    {"messages": [{"role": "user", "content": "..."}]}
    """
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    prompts.append(msg["content"])
    return prompts
