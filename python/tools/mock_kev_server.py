"""Local test double for a Kev System One server.

This is NOT a model. It reproduces the HTTP surface documented by the official
Kev project so the LaneShift adapter, the WebSocket bridge and the telemetry
path can be exercised end to end without downloading model weights.

It answers every choice question with a deterministic, valid option and echoes
the received request to a transcript file for inspection.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE_LOCK = threading.Lock()
TRANSCRIPT: list[dict] = []


class KevHandler(BaseHTTPRequestHandler):
    model_alias = "kev-latest"
    # The real server accepts both names for the same checkpoint.
    model_names = ("kev-latest", "jev-latest")
    run_id = "jaredpalmer/kev-0.8b"
    base_model = "Qwen/Qwen3.5-0.8B-Base"
    transcript_path: Path | None = None

    def log_message(self, *_args):  # keep stdout clean for the runner
        return

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-typesafe-request-id", f"mock-{len(STATE_LOCK and TRANSCRIPT) + 1}")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            card = {
                "description": f"Kev pointer head on {self.base_model}, serving {self.run_id}",
                "release_date": "2026-01-01",
                "run": self.run_id,
                "base": self.base_model,
                "lora": "mock",
                "device": "cpu",
                "backend": "mock",
                "dtype": "float32",
                "temperature": 1.0,
                "max_state_tokens": 65536,
            }
            self._send_json({"models": [{"name": name, **card} for name in self.model_names]})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/systemone":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        started = time.perf_counter()
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid json"}, status=400)
            return
        question = (request.get("questions") or {}).get("next_action") or {}
        criteria = question.get("criteria") or {}
        options = list(criteria.keys())
        if not options:
            self._send_json({"error": "no choice options offered"}, status=400)
            return
        chosen = options[0]
        probabilities = {option: (0.9 if option == chosen else 0.1 / max(1, len(options) - 1)) for option in options}
        total = sum(probabilities.values())
        probabilities = {option: round(value / total, 4) for option, value in probabilities.items()}
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        response = {
            "model": request.get("model", self.model_alias),
            "answers": {
                "next_action": {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": probabilities[chosen],
                    "probabilities": probabilities,
                }
            },
            "usage": {"input_tokens": 128, "output_tokens": 64},
            "latency_ms": latency_ms,
        }
        with STATE_LOCK:
            TRANSCRIPT.append({"request": request, "response": response})
            if self.transcript_path is not None:
                with self.transcript_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(TRANSCRIPT[-1], separators=(",", ":")) + "\n")
        self._send_json(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--run-id", default="jaredpalmer/kev-0.8b")
    parser.add_argument("--model-alias", default="kev-latest")
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args(argv)

    KevHandler.run_id = args.run_id
    KevHandler.model_alias = args.model_alias
    KevHandler.transcript_path = args.transcript
    if args.transcript is not None:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        args.transcript.write_text("", encoding="utf-8")

    server = ThreadingHTTPServer((args.host, args.port), KevHandler)
    print(f"mock Kev service listening at http://{args.host}:{args.port} (run={args.run_id})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
