"""
Dataset shard management for distributed training.

Strategy:
  - Uses HuggingFace datasets (streamed) to avoid full-download on every worker
  - Coordinator pre-assigns shard_id to each worker
  - Worker skips to its shard via deterministic interleaving (every N-th sample)
  - Supports any text dataset on the HF Hub (default: openwebtext)

Shard assignment: worker with shard_id S out of N total shards processes
every sample where (sample_index % N == S). This guarantees non-overlapping,
exhaustive coverage without needing to physically split files.
"""
from __future__ import annotations
import os
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset


HF_DATASET = os.environ.get("HF_DATASET", "openwebtext")
HF_SPLIT = os.environ.get("HF_SPLIT", "train")


class ShardedTextDataset(IterableDataset):
    """
    Streams text from HuggingFace Hub, selects every N-th document
    where N == total_shards and offset == shard_id.

    Tokenizes on-the-fly and yields fixed-length chunks of token IDs.
    """

    def __init__(
        self,
        shard_id: int,
        total_shards: int,
        seq_len: int,
        tokenizer,
        dataset_name: str = HF_DATASET,
        split: str = HF_SPLIT,
        buffer_tokens: int = 100_000,
    ):
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.buffer_tokens = buffer_tokens

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        token_buffer: list[int] = []
        doc_index = 0

        for sample in dataset:
            if doc_index % self.total_shards != self.shard_id:
                doc_index += 1
                continue
            doc_index += 1

            text = sample.get("text", "")
            if not text:
                continue

            ids = self.tokenizer.encode(text)
            token_buffer.extend(ids)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def get_tokenizer(vocab_size: int = 50_257):
    """Returns a tiktoken GPT-2 tokenizer (fast, no HF dependency for tokenization)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        raise ImportError("Install tiktoken: pip install tiktoken")


class _TiktokenWrapper:
    """Thin wrapper so tiktoken enc works with .encode() returning a list of ints."""
    def __init__(self, enc):
        self._enc = enc

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


def build_dataloader(
    shard_id: int,
    total_shards: int,
    seq_len: int,
    batch_size: int,
    dataset_name: str = HF_DATASET,
) -> torch.utils.data.DataLoader:
    enc = get_tokenizer()
    tokenizer = _TiktokenWrapper(enc)
    ds = ShardedTextDataset(
        shard_id=shard_id,
        total_shards=total_shards,
        seq_len=seq_len,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
