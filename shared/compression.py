"""
Gradient compression pipeline:
  1. Flatten tensors → float32 numpy array
  2. INT8 quantize  (scale per-tensor)
  3. gzip compress
  4. base64-encode for HTTP transport

decompress_gradients returns numpy arrays (no torch dependency)
so the coordinator can run without torch installed.
The aggregator converts to torch tensors only when needed.
"""
from __future__ import annotations
import gzip
import base64
import struct
import numpy as np
from typing import Dict, List, Tuple


def _pack_header(shapes: List[Tuple[int, ...]], names: List[str]) -> bytes:
    parts = []
    for name, shape in zip(names, shapes):
        name_b = name.encode("utf-8")
        parts.append(struct.pack(">H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack(">B", len(shape)))
        parts.extend(struct.pack(">I", s) for s in shape)
    return b"".join(parts)


def _unpack_header(data: bytes) -> Tuple[List[str], List[Tuple[int, ...]], int]:
    offset = 0
    names, shapes = [], []
    while offset < len(data):
        name_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
        name = data[offset:offset + name_len].decode("utf-8"); offset += name_len
        ndim = struct.unpack_from(">B", data, offset)[0]; offset += 1
        shape = tuple(struct.unpack_from(">I", data, offset + i * 4)[0] for i in range(ndim))
        offset += ndim * 4
        names.append(name)
        shapes.append(shape)
    return names, shapes, offset


def compress_gradients(gradients: Dict) -> bytes:
    """Compress named gradient tensors. Accepts torch tensors or numpy arrays."""
    names = list(gradients.keys())
    arrays = []
    for n in names:
        t = gradients[n]
        if hasattr(t, "detach"):
            arr = t.detach().float().cpu().numpy().flatten().astype(np.float32)
        else:
            arr = np.asarray(t, dtype=np.float32).flatten()
        arrays.append(arr)

    shapes = [gradients[n].shape if hasattr(gradients[n], "shape") else np.asarray(gradients[n]).shape for n in names]

    flat_arrays: List[np.ndarray] = []
    scales: List[float] = []
    for arr in arrays:
        scale = float(np.abs(arr).max()) / 127.0 + 1e-8
        quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        flat_arrays.append(quantized)
        scales.append(scale)

    header = _pack_header(shapes, names)
    header_len = struct.pack(">I", len(header))
    scales_bytes = np.array(scales, dtype=np.float32).tobytes()
    scales_len = struct.pack(">I", len(scales_bytes))
    payload = np.concatenate(flat_arrays).tobytes()

    raw = header_len + header + scales_len + scales_bytes + payload
    return base64.b64encode(gzip.compress(raw, compresslevel=6))


def decompress_gradients(data: bytes) -> Dict[str, np.ndarray]:
    """Decompress gradients. Returns numpy arrays — no torch required."""
    raw = gzip.decompress(base64.b64decode(data))
    offset = 0

    header_len = struct.unpack_from(">I", raw, offset)[0]; offset += 4
    header_bytes = raw[offset:offset + header_len]; offset += header_len
    names, shapes, _ = _unpack_header(header_bytes)

    scales_len = struct.unpack_from(">I", raw, offset)[0]; offset += 4
    scales = np.frombuffer(raw[offset:offset + scales_len], dtype=np.float32).tolist()
    offset += scales_len

    result: Dict[str, np.ndarray] = {}
    for name, shape, scale in zip(names, shapes, scales):
        numel = int(np.prod(shape))
        quantized = np.frombuffer(raw[offset:offset + numel], dtype=np.int8).copy()
        offset += numel
        result[name] = (quantized.astype(np.float32) * scale).reshape(shape)

    return result


def compute_pseudo_gradients(initial_params: Dict, final_params: Dict) -> Dict:
    """DiLoCo pseudo-gradient: delta = initial_weights - final_weights."""
    result = {}
    for name in initial_params:
        a = initial_params[name]
        b = final_params[name]
        if hasattr(a, "float"):
            result[name] = a.float() - b.float()
        else:
            result[name] = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return result
