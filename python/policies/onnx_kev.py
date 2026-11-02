"""Native ONNX Kev policy.

This adapter runs the exported Kev-0.8B decision graph with onnxruntime only. It
does not import torch or load the Kev repository. The prompt encoding mirrors
kev.model.encode: one state token, the rendered state, then one question branch
with each option span and a final decide token.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from kev_state import (
    ACTION_CRITERIA,
    ACTION_CRITERIA_LANE,
    ACTION_CRITERIA_SCENE,
    KEV_COMPACT_INSTRUCTIONS,
    KEV_LANE_INSTRUCTIONS,
    KEV_SCENE_INSTRUCTIONS,
    compact_kev_state,
    lane_observation,
    render_compact_state,
    render_lane_state,
    render_scene_state,
    scene_observation,
    scene_prompt_metadata,
)
from policy import PolicyDecision

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _user_tokens(tok: Tokenizer, text: str) -> list[int]:
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tok.encode(safe, add_special_tokens=False).ids


class OnnxKevPolicy:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        length: int = 256,
        intra_threads: int | None = None,
        data_path: str | None = None,
        arena: bool = False,
        model_id: str = "",
        label_source: str = "original_model",
        prompt_mode: str = "compact",
        providers: list[str] | None = None,
    ):
        self.model_path = str(Path(model_path))
        self.tokenizer_path = str(Path(tokenizer_path))
        self.data_path = str(Path(data_path)) if data_path else ""
        self.length = int(length)
        self.intra_threads = int(intra_threads or os.cpu_count() or 4)
        self.arena = bool(arena)
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.tok: Tokenizer | None = None
        self.session: ort.InferenceSession | None = None
        self.label_source = label_source if label_source in {"original_model", "fine_tuned_model"} else "original_model"
        self.policy_name = "onnx_kev"
        self.model_id = model_id or "jaredpalmer/kev-0.8b@v1.0+onnx"
        self.model_alias = "kev-onnx"
        self.model_sha256 = ""
        self.prompt_mode = prompt_mode if prompt_mode in {"compact", "lane", "scene"} else "compact"
        self.state_token_limit = max(8, self.length - 96)
        self.last_encoding: dict[str, Any] = {}
        self.load_ms = 0.0
        self.warmup_ms = 0.0
        self.ort_version = ort.__version__
        self._blob = None

    def _session_options(self, conservative: bool = False) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = (
            min(4, max(1, int(self.intra_threads))) if conservative else max(1, int(self.intra_threads))
        )
        # The graph is a single transformer pass. Inter-op parallelism only adds
        # scheduling overhead and memory pressure.
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = bool(self.arena) and not conservative
        if conservative:
            options.enable_mem_pattern = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return options

    def _checked_providers(self) -> list[str]:
        available = set(ort.get_available_providers())
        missing = [provider for provider in self.providers if provider not in available]
        if missing:
            raise RuntimeError(
                f"requested ONNX Runtime providers {missing} are not available; "
                f"this build offers {sorted(available)}"
            )
        return list(self.providers)

    async def start(self) -> None:
        self.tok = Tokenizer.from_file(self.tokenizer_path)
        self.model_sha256 = _file_sha256(self.model_path)
        blob_path = Path(self.data_path) if self.data_path else Path(self.model_path).with_name("kev07b_fp16.data")
        last_error: Exception | None = None
        drive = Path(self.model_path).drive.upper() if Path(self.model_path).drive else ""
        providers = self._checked_providers()
        # Some RAM-disk filesystems cannot answer the canonical-path query ONNX
        # Runtime uses for external data. Go straight to the in-memory loader on
        # those drives; a failed normal attempt leaves enough fragmentation to
        # make the fallback fail with bad allocation.
        if drive != "R:":
            options = self._session_options()
            try:
                started = time.perf_counter()
                self.session = ort.InferenceSession(self.model_path, options, providers=providers)
                self.load_ms = (time.perf_counter() - started) * 1000.0
            except Exception as first_error:
                last_error = first_error
            else:
                self._blob = None
                self._log_runtime(providers)
                # A warm-up failure is a real model/runtime failure, not a
                # reason to retry the whole load through the RAM-disk path.
                await self._warmup_async()
                return

        # In-memory external data path for RAM disks and other filesystems that
        # reject canonical path validation. Conservative options avoid the
        # constant-folding peak that causes a second bad allocation.
        if blob_path.exists():
            try:
                options = self._session_options(conservative=True)
                blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
                options.add_external_initializers_from_files_in_memory([blob_path.name], [blob], [blob.nbytes])
                proto = Path(self.model_path).read_bytes()
                started = time.perf_counter()
                self.session = ort.InferenceSession(proto, options, providers=providers)
                self.load_ms = (time.perf_counter() - started) * 1000.0
            except Exception as fallback_error:
                last_error = fallback_error
            else:
                self._blob = blob
                print("onnx_kev: loaded external weights through the in-memory fallback", flush=True)
                self._log_runtime(providers)
                await self._warmup_async()
                return
        raise RuntimeError(f"could not load ONNX model {self.model_path}: {last_error}") from last_error

    def _log_runtime(self, providers: list[str]) -> None:
        print(
            f"onnx_kev: onnxruntime {self.ort_version} providers={providers} "
            f"intra_threads={self.intra_threads} arena={bool(self.arena)} "
            f"session_load_ms={self.load_ms:.0f}",
            flush=True,
        )

    def _warmup_state(self) -> dict[str, Any]:
        return {
            "episode_id": "warmup",
            "sim_time": 0.0,
            "physics_tick": 0,
            "player": {
                "lane": 1,
                "target_lane": 1,
                "lane_change_in_progress": False,
                "lane_change_cooldown": 0.0,
                "speed": 20.0,
                "base_speed": 20.0,
                "max_speed": 28.0,
                "acceleration": 3.0,
                "braking": False,
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "current_action": "STAY",
            },
            "traffic": [],
            "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
        }

    def warmup(self) -> float:
        """Pay tokenizer, graph, and schedule costs before the first real decision."""
        started = time.perf_counter()
        self._infer(self._warmup_state())
        self.warmup_ms = (time.perf_counter() - started) * 1000.0
        return self.warmup_ms

    async def _warmup_async(self) -> None:
        # Nothing is connected yet, so running the warm-up off the event loop is
        # about keeping the server responsive, not about gameplay.
        await asyncio.to_thread(self.warmup)
        print(f"onnx_kev: warmup_ms={self.warmup_ms:.0f}", flush=True)

    def _encode(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        if self.tok is None:
            raise RuntimeError("ONNX Kev policy has not started")
        compact = compact_kev_state(state)
        legal = [a for a in compact["legal_actions"] if a in ACTION_CRITERIA]
        if not legal:
            raise ValueError("no legal actions to offer the model")
        prompt_meta: dict[str, int] = {}
        if self.prompt_mode == "scene":
            observation = scene_observation(state)
            state_text = render_scene_state(observation)
            instructions = KEV_SCENE_INSTRUCTIONS
            criteria = ACTION_CRITERIA_SCENE
            prompt_meta = scene_prompt_metadata(observation)
        elif self.prompt_mode == "lane":
            state_text = render_lane_state(lane_observation(state))
            instructions = KEV_LANE_INSTRUCTIONS
            criteria = ACTION_CRITERIA_LANE
        else:
            state_text = render_compact_state(compact)
            instructions = KEV_COMPACT_INSTRUCTIONS
            criteria = ACTION_CRITERIA
        q_id = self.tok.token_to_id(SPECIAL[1])
        o_id = self.tok.token_to_id(SPECIAL[2])
        c_id = self.tok.token_to_id(SPECIAL[3])
        d_id = self.tok.token_to_id(SPECIAL[4])
        instr = [q_id] + _user_tokens(self.tok, instructions)
        spans = [[o_id] + _user_tokens(self.tok, f"{action}: {criteria[action]}") + [c_id] for action in legal]
        branch = instr + [token for span in spans for token in span] + [d_id]
        # The fixed graph leaves a finite state budget. Both renderers stay under
        # it; a future renderer must fail loudly instead of silently slicing off
        # traffic or the legal-action tail.
        max_state = max(8, self.length - len(branch))
        state_tokens = _user_tokens(self.tok, state_text)
        if len(state_tokens) + 1 > max_state:
            raise ValueError(
                f"{self.prompt_mode} observation needs {len(state_tokens) + 1} state tokens; "
                f"only {max_state} fit without dropping information"
            )
        state_ids = [self.tok.token_to_id(SPECIAL[0])] + state_tokens
        ids = list(state_ids)
        pos = list(range(len(state_ids)))
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
        self.last_encoding = {
            "prompt_mode": self.prompt_mode,
            "state_tokens": len(state_ids),
            "total_tokens": real,
            "state_truncated": False,
            "legal_actions": legal,
            "state_text": state_text,
            **prompt_meta,
        }
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
        encoded = time.perf_counter()
        if self.session is None:
            raise RuntimeError("ONNX Kev session is not loaded")
        logits = np.asarray(self.session.run(None, feeds)[0]).ravel().astype(np.float64)
        finished = time.perf_counter()
        if logits.size != len(self.last_encoding["legal_actions"]):
            raise ValueError(f"model returned {logits.size} logits for {len(self.last_encoding['legal_actions'])} options")
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        actions = self.last_encoding["legal_actions"]
        probabilities = {action: float(p[i]) for i, action in enumerate(actions)}
        action = max(probabilities, key=probabilities.get)
        self.last_encoding["encode_ms"] = round((encoded - started) * 1000.0, 2)
        self.last_encoding["run_ms"] = round((finished - encoded) * 1000.0, 2)
        self.last_encoding["total_ms"] = round((finished - started) * 1000.0, 2)
        return action, probabilities, self.last_encoding.copy(), (finished - started) * 1000.0

    def choose_action(self, state: dict[str, Any]) -> PolicyDecision:
        """Blocking inference.

        The controller runs every synchronous policy on one worker thread, so a
        timed-out decision can never overlap the next one on the same session.
        """
        started = time.perf_counter()
        action, probabilities, encoding, model_ms = self._infer(state)
        return PolicyDecision(
            action=action,
            label_source=self.label_source,
            policy_name=self.policy_name,
            model_id=self.model_id,
            model_alias=self.model_alias,
            model_sha256=self.model_sha256,
            probabilities=probabilities,
            confidence=probabilities[action],
            model_latency_ms=model_ms,
            client_latency_ms=(time.perf_counter() - started) * 1000.0,
            request_payload={
                "prompt_mode": self.prompt_mode,
                "state": encoding.get("state_text", ""),
                "encoding": encoding,
            },
            response_payload={"probabilities": probabilities, "encoding": encoding},
        )
