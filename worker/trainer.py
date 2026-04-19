"""
DiLoCo inner training loop using HuggingFace transformers + peft (LoRA).

Only LoRA adapter parameters are synced — base model weights never leave
the worker's machine. This makes uploads ~100x smaller than full fine-tuning.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from shared.compression import compress_gradients, compute_pseudo_gradients
from shared.protocol import TrainingConfig
from training.model import (
    get_lora_state_dict,
    load_model_and_tokenizer,
)
from worker.gpu_detect import HardwareProfile, to_torch_dtype

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    steps_completed: int
    final_loss: float
    compressed_gradients: bytes
    elapsed_seconds: float
    interrupted: bool = False


class LocalTrainer:
    def __init__(self, config: TrainingConfig, hw: HardwareProfile):
        self.config = config
        self.hw = hw
        self.device = torch.device(hw.torch_device)
        self.dtype = to_torch_dtype(hw.dtype)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
        shard_id: int = 0,
        total_shards: int = 1,
    ) -> TrainingResult:
        model, tokenizer = load_model_and_tokenizer(
            self.config.hf_id or self.config.model_name,
            self.config.lora_r,
            self.config.lora_alpha,
            self.config.lora_target_modules,
            self.device,
            self.dtype,
            adapter_path,
        )
        model.train()

        # Snapshot initial LoRA weights for pseudo-gradient computation
        initial_params = get_lora_state_dict(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        use_amp = self.device.type == "cuda" and self.dtype != torch.float32
        scaler = GradScaler(enabled=use_amp)

        dataloader = self._build_dataloader(dataset_path, tokenizer, shard_id, total_shards)
        data_iter = iter(dataloader)

        total_loss = 0.0
        steps = 0
        t0 = time.time()

        for step in range(self.config.local_steps):
            if self._stop_requested:
                logger.info("Training interrupted at step %d", step)
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, dtype=self.dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                self.config.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

            if step % 10 == 0:
                logger.info("Step %d/%d  loss=%.4f", step, self.config.local_steps, loss.item())

        avg_loss = total_loss / steps if steps > 0 else float("inf")
        elapsed = time.time() - t0

        final_params = get_lora_state_dict(model)
        pseudo_grads = compute_pseudo_gradients(initial_params, final_params)
        compressed = compress_gradients(pseudo_grads)

        logger.info("Done: %d steps, loss=%.4f, elapsed=%.1fs, upload=%.1fKB",
                    steps, avg_loss, elapsed, len(compressed) / 1024)

        return TrainingResult(
            steps_completed=steps,
            final_loss=avg_loss,
            compressed_gradients=compressed,
            elapsed_seconds=elapsed,
            interrupted=self._stop_requested,
        )

    def _build_dataloader(self, dataset_path: str, tokenizer, shard_id: int, total_shards: int):
        """Load JSONL dataset shard and return a DataLoader."""
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % total_shards != shard_id:
                    continue
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        if not records:
            raise ValueError(f"Shard {shard_id}/{total_shards} is empty in {dataset_path}")

        logger.info("Shard %d: %d records", shard_id, len(records))

        texts = [self._format_record(r) for r in records]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        dataset = _TokenDataset(encodings)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def _format_record(record: dict) -> str:
        """Convert JSONL record to training text. Supports multiple formats."""
        if "text" in record:
            return record["text"]
        if "instruction" in record:
            inp = record.get("input", "")
            out = record.get("output", "")
            if inp:
                return f"### Instruction:\n{record['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            return f"### Instruction:\n{record['instruction']}\n\n### Response:\n{out}"
        if "prompt" in record and "completion" in record:
            return record["prompt"] + record["completion"]
        return str(record)


class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        # mask padding tokens in labels
        item["labels"][item["labels"] == 0] = -100
        return item
