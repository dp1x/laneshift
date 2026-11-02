r"""Export one fixed-length Kev-0.8B graph, reusing the existing fp16 weight blob.

Run from the Kev repo with its venv:
  uv run python C:\Users\Dhane\dev\kev-onnx-backup\export_reuse.py 256
"""
import os
import sys
import time

import numpy as np
import onnx
import torch
from onnx.external_data_helper import set_external_data

BACKUP = r"C:\Users\Dhane\dev\kev-onnx-backup"
sys.path.insert(0, BACKUP)

from kev_onnx import Decide, load_model, patch_transformers_for_tracing, patch_triangular_solve, row_inputs
import inputs

REAL_INPUTS = ["input_ids", "position_ids", "attention_mask", "decide_idx", "opt_idx"]
DATA_NAME = "kev07b_fp16.data"
DTYPES = {np.dtype("float16"): 10, np.dtype("float32"): 1, np.dtype("int64"): 7, np.dtype("uint8"): 2}


def offsets_for(sd):
    offsets = {}
    cursor = 0
    for name, t in sd.items():
        arr = t.detach().cpu().contiguous().numpy()
        offsets[name] = (cursor, arr.nbytes)
        cursor += arr.nbytes
        cursor += (-arr.nbytes) % 64
    return offsets, cursor


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    patch_transformers_for_tracing()
    patch_triangular_solve(True)
    t0 = time.time()
    tok, m = load_model(dtype=torch.float16)
    print(f"loaded fp16 in {time.time() - t0:.1f}s", flush=True)
    enc = m.encode(tok, inputs.TINY)
    real = len(enc["ids"])
    if real > target:
        raise SystemExit(f"tiny record is {real} tokens, over the {target} target")
    ids = enc["ids"] + [0] * (target - real)
    pos = enc["pos"] + [0] * (target - real)
    mask = [1] * real + [0] * (target - real)
    print("row tokens", target, "real", real, flush=True)
    w = Decide(m).eval()
    args = (torch.tensor([ids], dtype=torch.long), torch.tensor([pos], dtype=torch.long),
            torch.tensor([mask], dtype=torch.long), torch.tensor([enc["decide_idx"][0]], dtype=torch.long),
            torch.tensor(enc["opt_idx"][0], dtype=torch.long))
    graph = os.path.join(BACKUP, f"graph_L{target}.onnx")
    with torch.no_grad():
        torch.onnx.export(w, args, graph, input_names=REAL_INPUTS, output_names=["logits"], opset_version=17,
                          do_constant_folding=False, dynamo=False, export_params=False,
                          keep_initializers_as_inputs=True,
                          dynamic_axes={"opt_idx": {0: "K"}, "logits": {0: "K"}})
    print("graph MB", round(os.path.getsize(graph) / 1e6, 2), flush=True)

    sd = w.state_dict()
    offsets, total = offsets_for(sd)
    blob_path = os.path.join(BACKUP, DATA_NAME)
    print("blob bytes expected", total, "actual", os.path.getsize(blob_path), flush=True)
    if total != os.path.getsize(blob_path):
        raise SystemExit("existing weight blob does not match the state dict layout")

    mo = onnx.load(graph, load_external_data=False)
    wnames = [i.name for i in mo.graph.input if i.name not in REAL_INPUTS]
    keep = [i for i in mo.graph.input if i.name in REAL_INPUTS]
    del mo.graph.input[:]
    mo.graph.input.extend(keep)
    for name in wnames:
        if name not in offsets:
            raise SystemExit(f"missing initializer {name}")
        off, nbytes = offsets[name]
        arr = sd[name].detach().cpu().contiguous().numpy()
        tp = onnx.TensorProto(name=name, data_type=DTYPES[arr.dtype], dims=list(arr.shape))
        tp.data_location = onnx.TensorProto.EXTERNAL
        for k, v in (("location", DATA_NAME), ("offset", off), ("length", nbytes)):
            entry = tp.external_data.add()
            entry.key, entry.value = k, str(v)
        mo.graph.initializer.append(tp)
    out = os.path.join(BACKUP, f"kev07b_fp16_L{target}.onnx")
    onnx.save_model(mo, out)
    print("wrote", out, round(os.path.getsize(out) / 1e6, 2), "MB", flush=True)


if __name__ == "__main__":
    main()
