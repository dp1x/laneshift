"""Export Kev-0.8B's decision pass to ONNX without ever materialising a weight-carrying ModelProto.

torch.onnx's exporters both build a proto (or FX copy) holding all 1.5 GB of fp16 weights, and this VM has no pagefile:
every attempt died with MemoryError. So: export the graph with `export_params=False` (weights become graph inputs, the
proto stays ~5 MB), then stream the tensors straight from the module into one external-data file both graphs share.
"""
import os
import sys
import time

import numpy as np
import onnx
import torch

from kev_onnx import Decide, load_model, patch_transformers_for_tracing, patch_triangular_solve, row_inputs
import inputs

REAL_INPUTS = ["input_ids", "position_ids", "attention_mask", "decide_idx", "opt_idx"]
OUT = r"R:\Temp\kev-onnx"
DATA_NAME = "kev07b_fp16.data"
DTYPES = {np.dtype("float16"): 10, np.dtype("float32"): 1, np.dtype("int64"): 7, np.dtype("uint8"): 2}


def export_graph(w, args, path):
    with torch.no_grad():
        torch.onnx.export(w, args, path, input_names=REAL_INPUTS, output_names=["logits"], opset_version=17,
                          do_constant_folding=False, dynamo=False, export_params=False,
                          keep_initializers_as_inputs=True)
    return path


def build_data(sd, data_path):
    """One fp16/fp32 blob for every parameter, 64-byte aligned; returns name -> (offset, nbytes)."""
    offsets = {}
    with open(data_path, "wb") as f:
        for name, t in sd.items():
            arr = t.detach().cpu().contiguous().numpy()
            off = f.tell()
            arr.tofile(f)
            offsets[name] = (off, arr.nbytes)
            pad = (-arr.nbytes) % 64
            if pad:
                f.write(b"\0" * pad)
    return offsets


def attach(graph_path, out_path, sd, offsets, data_name):
    mo = onnx.load(graph_path, load_external_data=False)
    wnames = [i.name for i in mo.graph.input if i.name not in REAL_INPUTS]
    unknown = [n for n in wnames if n not in offsets]
    if unknown:
        raise RuntimeError(f"{len(unknown)} graph inputs have no parameter, e.g. {unknown[:3]}")
    keep = [i for i in mo.graph.input if i.name in REAL_INPUTS]
    del mo.graph.input[:]
    mo.graph.input.extend(keep)
    for name in wnames:
        off, nbytes = offsets[name]
        arr = sd[name].detach().cpu().contiguous().numpy()
        tp = onnx.TensorProto(name=name, data_type=DTYPES[arr.dtype], dims=list(arr.shape))
        tp.data_location = onnx.TensorProto.EXTERNAL
        for k, v in (("location", data_name), ("offset", off), ("length", nbytes)):
            entry = tp.external_data.add()
            entry.key, entry.value = k, str(v)
        mo.graph.initializer.append(tp)
    onnx.save_model(mo, out_path)
    return out_path


def main():
    patch_transformers_for_tracing()
    t0 = time.time()
    tok, m = load_model(dtype=torch.float16)
    print(f"loaded fp16 in {time.time() - t0:.1f}s", flush=True)
    w = Decide(m).eval()
    sd = w.state_dict()

    cases = {}
    for target in (35, 585):
        enc = m.encode(tok, inputs.TINY) if target <= 100 else inputs.build(m, tok, target)
        L = len(enc["ids"])
        args = row_inputs(enc)
        with torch.no_grad():                          # reference BEFORE any patch: the real solve_triangular
            z = w(*args).float()
        np.savez(os.path.join(OUT, f"ref_L{L}.npz"),
                 ids=args[0].numpy(), pos=args[1].numpy(), mask=args[2].numpy(),
                 dec=args[3].numpy(), opt=args[4].numpy(),
                 logits=z.numpy().ravel(), probs=torch.softmax(z, -1).numpy().ravel())
        print(f"L={L}: torch logits {z.tolist()} finite={bool(torch.isfinite(z).all())}", flush=True)
        patch_triangular_solve(True)
        t = time.time()
        g = export_graph(w, args, os.path.join(OUT, f"graph_L{L}.onnx"))
        patch_triangular_solve(False)
        print(f"L={L}: graph exported in {time.time() - t:.1f}s, {os.path.getsize(g) / 1e6:.1f} MB", flush=True)
        cases[L] = g

    del m, w  # the graphs hold no weights; the module must go before the blob is written twice over
    import gc
    gc.collect()

    t = time.time()
    offsets = build_data(sd, os.path.join(OUT, DATA_NAME))
    print(f"data blob {os.path.getsize(os.path.join(OUT, DATA_NAME)) / 1e6:.0f} MB in {time.time() - t:.1f}s", flush=True)

    for L, g in cases.items():
        out = attach(g, os.path.join(OUT, f"kev07b_fp16_L{L}.onnx"), sd, offsets, DATA_NAME)
        print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
