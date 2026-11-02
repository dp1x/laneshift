"""ONNX-exportable wrapper around one Kev decision (read-only use of the kev package)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.checkpoint import Checkpoint, LoadOptions

RUN = "jaredpalmer/kev-0.8b@v1.0"


def patch_transformers_for_tracing():
    """transformers 5.17's causal_conv1d_fn derives `padding` via `weight.shape[-1] - 1`, which under torch.jit.trace is a
    0-dim tensor, not an int, so F.conv1d rejects it. Replace it with an int-equivalent (same math, same weights)."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35
    act2fn = q35.ACT2FN

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        seq_len = hidden_states.shape[-1]
        hidden_size = hidden_states.shape[1]
        padding = int(weight.shape[-1].item()) - 1 if isinstance(weight.shape[-1], torch.Tensor) else int(weight.shape[-1]) - 1
        out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
                       padding=padding, groups=hidden_size)[:, :, :seq_len]
        if activation is not None:
            out = act2fn[activation](out)
        return out.to(hidden_states.dtype)

    q35.causal_conv1d_fn = causal_conv1d_fn


_ORIG_SOLVE_TRIANGULAR = torch.linalg.solve_triangular


def _blocked_triangular_solve(A, B, upper=False, unitriangular=False, **kwargs):
    """Solve A X = B for a lower unit-triangular A (the only case Qwen3.5's gated delta rule uses) with ops ONNX has.

    (I+N)^-1 for the strictly lower N, blocked 16x16: each diagonal block is inverted with the finite Neumann product
    (I-D)(I+D^2)(I+D^4)(I+D^8) (exact, D^16 = 0), then the off-diagonal blocks by block forward substitution. A plain
    64-step forward substitution is exact too but unrolls into ~20k ONNX nodes per export; a whole-matrix Neumann series
    is not usable (fp32 overflow at D^32)."""
    if upper or not unitriangular:
        return _ORIG_SOLVE_TRIANGULAR(A, B, upper=upper, unitriangular=unitriangular, **kwargs)
    c = A.shape[-1]
    b = 16 if c % 16 == 0 and c >= 16 else 8
    m = c // b
    N = A.tril(-1)
    eye = torch.eye(b, dtype=A.dtype, device=A.device)

    def inv_small(D):                      # (I + D)^-1 for strictly lower D of size b
        X, P = eye - D, D
        for _ in range(b.bit_length() - 2):
            P = P @ P
            X = X @ (eye + P)
        return X

    diag = [N[..., i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(m)]
    inv_diag = [inv_small(d) for d in diag]
    blocks = [[None] * m for _ in range(m)]
    for i in range(m):
        blocks[i][i] = inv_diag[i]
        for j in range(i):
            acc = N[..., i * b:(i + 1) * b, j * b:(j + 1) * b] @ blocks[j][j]
            for k in range(j + 1, i):
                acc = acc + N[..., i * b:(i + 1) * b, k * b:(k + 1) * b] @ blocks[k][j]
            blocks[i][j] = -(inv_diag[i] @ acc)
    zero = torch.zeros_like(blocks[0][0])
    for i in range(m):
        for j in range(i + 1, m):
            blocks[i][j] = zero
    X = torch.cat([torch.cat(blocks[i], dim=-1) for i in range(m)], dim=-2)
    return X @ B


def patch_triangular_solve(enable=True):
    """torch.onnx's TorchScript exporter has no symbolic for aten::linalg_solve_triangular. Swapped in for export only;
    _ORIG_SOLVE_TRIANGULAR keeps the reference path untouched."""
    torch.linalg.solve_triangular = _blocked_triangular_solve if enable else _ORIG_SOLVE_TRIANGULAR


def load_model(dtype=torch.float16, merge=True):
    ck = Checkpoint(RUN)
    return ck.load("cpu", LoadOptions(dtype=dtype, merge=merge))


class Decide(nn.Module):
    """One causal row (state + one branch) -> option logits [K].

    Mirrors DecisionModel._rows_hidden + PointerHead.forward for a single row without padding:
    input_ids/position_ids/attention_mask are [1, L], decide_idx [1], opt_idx [K]."""

    def __init__(self, m):
        super().__init__()
        self.lm = m.lm
        self.head = m.head

    def forward(self, input_ids, position_ids, attention_mask, decide_idx, opt_idx):
        hs = self.lm(input_ids=input_ids, position_ids=position_ids,
                     attention_mask=attention_mask, use_cache=False).last_hidden_state
        h = hs[0].float()
        return self.head(h.index_select(0, decide_idx)[0], h.index_select(0, opt_idx))


def row_inputs(enc, torch_mod=torch):
    """(ids, pos, mask, decide_idx, opt_idx) tensors for a single-row encoding, as the wrapper takes them."""
    ids = torch_mod.tensor([enc["ids"]], dtype=torch.long)
    pos = torch_mod.tensor([enc["pos"]], dtype=torch.long)
    att = torch_mod.ones_like(ids)
    return ids, pos, att, torch_mod.tensor([enc["decide_idx"][0]]), torch_mod.tensor(enc["opt_idx"][0])
