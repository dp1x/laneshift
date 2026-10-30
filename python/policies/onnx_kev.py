"""Native ONNX Kev policy.

This adapter runs the exported Kev-0.8B decision graph with onnxruntime only. It
does not import torch or load the Kev repository. The prompt encoding mirrors
kev.model.encode: one state token, the rendered state, then one question branch
with each option span and a final decide token.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from kev_state import ACTION_CRITERIA, compact_kev_state
from policy import PolicyDecision

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
INSTRUCTIONS = "Choose the safest useful next action from the currently legal choices. Only select an option present in criteria."


def _render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {_render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        f"{pad}{k}:\n{_render(v, indent + 1)}" if isinstance(v, (dict, list)) else f"{pad}{k}: {_render(v)}"
        for k, v in value.items()
    )


def _user_tokens(tok: Tokenizer, text: str) -> list[int]:
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tok.encode(safe, add_special_tokens=False).ids


class OnnxKevPolicy:
    def __init__(self, model_path: str, tokenizer_path: str, length: int = 256, intra_threads: int | None = None):
        self.model_path = str(Path(model_path))
        self.tokenizer_path = str(Path(tokenizer_path))
        self.length = int(length)
        self.intra_threads = int(intra_threads or os.cpu_count() or 4)
        self.tok: Tokenizer | None = None
        self.session: ort.InferenceSession | None = None
        self.label_source = "original_model"
        self.policy_name = "onnx_kev"
        self.model_id = "jaredpalmer/kev-0.8b@v1.0+onnx-fp16"
        self.model_alias = "kev-onnx"
        self.state_token_limit = max(8, self.length - 96)
        self.last_encoding: dict[str, Any] = {}

    async def start(self) -> None:
        self.tok = Tokenizer.from_file(self.tokenizer_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = self.intra_threads
        blob_path = Path(self.model_path).with_name("kev07b_fp16.data")
        if blob_path.exists():
            blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
            options.add_external_initializers_from_files_in_memory([blob_path.name], [blob], [blob.nbytes])
            proto = Path(self.model_path).read_bytes()
            self.session = ort.InferenceSession(proto, options, providers=["CPUExecutionProvider"])
            self._blob = blob
        else:
            self.session = ort.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])

    def _encode(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        if self.tok is None:
            raise RuntimeError("ONNX Kev policy has not started")
        compact = compact_kev_state(state)
        legal = [a for a in compact["legal_actions"] if a in ACTION_CRITERIA]
        if not legal:
            raise ValueError("no legal actions to offer the model")
        state_text = _render(compact)
        state_tokens = _user_tokens(self.tok, state_text)
        if len(state_tokens) + 1 > self.state_token_limit:
            state_tokens = state_tokens[: self.state_token_limit - 1]
        state_ids = [self.tok.token_to_id(SPECIAL[0])] + state_tokens
        ids = list(state_ids)
        pos = list(range(len(state_ids)))
        q_id = self.tok.token_to_id(SPECIAL[1])
        o_id = self.tok.token_to_id(SPECIAL[2])
        c_id = self.tok.token_to_id(SPECIAL[3])
        d_id = self.tok.token_to_id(SPECIAL[4])
        instr = [q_id] + _user_tokens(self.tok, INSTRUCTIONS)
        spans = [[o_id] + _user_tokens(self.tok, f"{action}: {ACTION_CRITERIA[action]}") + [c_id] for action in legal]
        branch = instr + [token for span in spans for token in span] + [d_id]
        base = len(ids)
        ids += branch
        pos += list(range(base, base + len(branch)))
        decide_idx = base + len(branch) - 1
        ends, cursor = [], len(instr)
        for span in spans:
            cursor += len(span)
            ends.append(base + cursor - 1)
        real = len(ids)
        if real > self.length:
            raise ValueError(f"Kev prompt is {real} tokens, over the {self.length}-token ONNX graph")
        pad = self.length - real
        input_ids = np.asarray([ids + [0] * pad], dtype=np.int64)
        position_ids = np.asarray([pos + [0] * pad], dtype=np.int64)
        attention_mask = np.asarray([[1] * real + [0] * pad], dtype=np.int64)
        decide = np.asarray([decide_idx], dtype=np.int64)
        option_idx = np.asarray(ends, dtype=np.int64)
        self.last_encoding = {"state_tokens": len(state_ids), "total_tokens": real, "legal_actions": legal}
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "decide_idx": decide,
            "opt_idx": option_idx,
        }

    def _infer(self, state: dict[str, Any]) -> tuple[str, dict[str, float], dict[str, Any], float]:
        started = time.perf_counter()
        feeds = self._encode(state)
        if self.session is None:
            raise RuntimeError("ONNX Kev session is not loaded")
        logits = np.asarray(self.session.run(None, feeds)[0]).ravel().astype(np.float64)
        if logits.size != len(self.last_encoding["legal_actions"]):
            raise ValueError(f"model returned {logits.size} logits for {len(self.last_encoding['legal_actions'])} options")
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        actions = self.last_encoding["legal_actions"]
        probabilities = {action: float(p[i]) for i, action in enumerate(actions)}
        action = max(probabilities, key=probabilities.get)
        return action, probabilities, self.last_encoding.copy(), (time.perf_counter() - started) * 1000.0

    async def choose_action(self, state: dict[str, Any]) -> PolicyDecision:
        started = time.perf_counter()
        action, probabilities, encoding, model_ms = await asyncio.to_thread(self._infer, state)
        return PolicyDecision(
            action=action,
            label_source=self.label_source,
            policy_name=self.policy_name,
            model_id=self.model_id,
            model_alias=self.model_alias,
            probabilities=probabilities,
            confidence=probabilities[action],
            model_latency_ms=model_ms,
            client_latency_ms=(time.perf_counter() - started) * 1000.0,
            request_payload={"state": compact_kev_state(state), "encoding": encoding},
            response_payload={"probabilities": probabilities, "encoding": encoding},
        )
