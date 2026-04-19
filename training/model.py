"""
HuggingFace model loader with LoRA (via peft).

Base model weights are pulled from HF Hub and cached locally on the worker.
Only LoRA adapter parameters (~1-5% of total params) are ever synced
over the network — this is what makes internet-distributed training viable.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    hf_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_target_modules: List[str],
    device: torch.device,
    dtype: torch.dtype,
    adapter_path: Optional[str] = None,
):
    """
    Load a base model from HF Hub, apply LoRA, optionally load existing adapter.
    Returns (model, tokenizer).
    """
    logger.info("Loading base model: %s", hf_id)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # CPU workers load in float32; GPU workers use device_map for multi-GPU support
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["device_map"] = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    base_model.config.use_cache = False  # required for gradient checkpointing

    if adapter_path and Path(adapter_path).exists():
        # Resume from existing LoRA adapter
        logger.info("Loading existing LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    else:
        # Fresh LoRA adapter
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model, tokenizer


def save_adapter(model, path: str) -> None:
    """Save only the LoRA adapter weights (not the full base model)."""
    model.save_pretrained(path)
    logger.info("LoRA adapter saved to %s", path)


def get_lora_state_dict(model) -> dict[str, torch.Tensor]:
    """Extract only the trainable LoRA parameter tensors."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_lora_state_dict(model, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameter tensors into a model in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in state_dict:
                param.copy_(state_dict[name].to(param.device))
