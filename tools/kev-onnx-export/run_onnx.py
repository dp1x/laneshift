"""Run one exported ONNX file with onnxruntime only (no torch), check parity against the saved PyTorch reference and
measure warm latency:  python run_onnx.py <length> <level>     (level: none | basic | extended | all)

The external-data blob lives next to this script. It is handed to the session from a read-only memory map rather than
by path: on some filesystems (the R: RAM disk this was built on) GetFinalPathNameByHandle fails and both onnx and
onnxruntime refuse to open an external-data file by path ("path could not be canonicalized: Incorrect function").
A memory map works on every filesystem, so this is the portable path.
"""
import os
import platform
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_NAME = "kev07b_fp16.data"
LEVELS = {"none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
          "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
          "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def main(length, level, runs=3):
    ref = np.load(os.path.join(HERE, f"ref_L{length}.npz"))
    so = ort.SessionOptions()
    so.graph_optimization_level = LEVELS[level]
    blob = np.memmap(os.path.join(HERE, DATA_NAME), dtype=np.uint8, mode="r")
    so.add_external_initializers_from_files_in_memory([DATA_NAME], [blob], [blob.nbytes])
    t0 = time.perf_counter()
    sess = ort.InferenceSession(open(os.path.join(HERE, f"kev07b_fp16_L{length}.onnx"), "rb").read(), so,
                                providers=["CPUExecutionProvider"])
    init = time.perf_counter() - t0
    print(f"platform.machine()={platform.machine()} python={sys.version.split()[0]} {platform.architecture()[0]}")
    print(f"onnxruntime {ort.__version__} device={ort.get_device()} providers={ort.get_available_providers()}")
    print(f"L={length} level={level} session init {init:.1f}s  inputs={[i.name for i in sess.get_inputs()]}")
    feeds = {"input_ids": ref["ids"], "position_ids": ref["pos"], "attention_mask": ref["mask"],
             "decide_idx": ref["dec"], "opt_idx": ref["opt"]}
    z = np.asarray(sess.run(None, feeds)[0]).ravel().astype(np.float64)
    e = np.exp(z - z.max())
    p = e / e.sum()
    print("logits ort  ", np.round(z, 6), "| torch", np.round(ref["logits"], 6))
    print("probs  ort  ", np.round(p, 6), "| torch", np.round(ref["probs"], 6))
    print(f"max |dlogit| {np.abs(z - ref['logits']).max():.3e} | max |dprob| {np.abs(p - ref['probs']).max():.3e}")
    t = []
    for i in range(runs + 1):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        dt = (time.perf_counter() - t0) * 1000
        if i:
            t.append(dt)
    print(f"latency ms {[round(x, 1) for x in t]} -> median {statistics.median(t):.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 35, sys.argv[2] if len(sys.argv) > 2 else "basic")
