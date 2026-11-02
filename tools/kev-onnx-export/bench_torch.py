"""PyTorch reference latency for the same two inputs, on the same machine: uv run python bench_torch.py
(fp16 because that is the dtype the ONNX file carries!)"""
import statistics
import time

import torch

from kev_onnx import Decide, load_model, patch_transformers_for_tracing, row_inputs
import inputs

patch_transformers_for_tracing()
tok, m = load_model(dtype=torch.float16)
w = Decide(m).eval()
print("torch", torch.__version__, "threads", torch.get_num_threads(), "dtype", m.dtype)
for name, target in (("tiny", 35), ("12veh", 585)):
    enc = m.encode(tok, inputs.TINY) if target <= 100 else inputs.build(m, tok, target)
    args = row_inputs(enc)
    with torch.no_grad():
        w(*args)                       # warm
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            z = w(*args)
            t.append((time.perf_counter() - t0) * 1000)
    print(f"{name} L={len(enc['ids'])}: {[round(x, 1) for x in t]} -> median {statistics.median(t):.1f} ms; "
          f"logits {[round(v, 6) for v in z.tolist()]}")
