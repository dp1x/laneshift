import os
import time
import numpy as np
import onnxruntime as ort

BACKUP = r"C:\Users\Dhane\dev\kev-onnx-backup"
L = 256
blob = np.memmap(os.path.join(BACKUP, "kev07b_fp16.data"), dtype=np.uint8, mode="r")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.add_external_initializers_from_files_in_memory(["kev07b_fp16.data"], [blob], [blob.nbytes])
proto = open(os.path.join(BACKUP, f"kev07b_fp16_L{L}.onnx"), "rb").read()
sess = ort.InferenceSession(proto, so, providers=["CPUExecutionProvider"])
ref = np.load(os.path.join(BACKUP, "ref_L35.npz"))
ids = ref["ids"]
pos = ref["pos"]
mask = ref["mask"]
real = ids.shape[1]
pad = L - real
feeds = {
    "input_ids": np.pad(ids, ((0, 0), (0, pad)), constant_values=0),
    "position_ids": np.pad(pos, ((0, 0), (0, pad)), constant_values=0),
    "attention_mask": np.pad(mask, ((0, 0), (0, pad)), constant_values=0),
    "decide_idx": ref["dec"],
    "opt_idx": ref["opt"],
}
z = None
for i in range(4):
    t0 = time.perf_counter()
    z = np.asarray(sess.run(None, feeds)[0]).ravel().astype(np.float64)
    print(f"run{i} ms={(time.perf_counter() - t0) * 1000:.1f}")
e = np.exp(z - z.max())
p = e / e.sum()
print("logits", np.round(z, 6), "ref", np.round(ref["logits"], 6))
print("probs", np.round(p, 6), "ref", np.round(ref["probs"], 6))
print("max dprob", np.abs(p - ref["probs"]).max())
