# Kev GPU benchmark notebook

`kev_gpu_benchmark.ipynb` measures warm **single-request** latency for
`jaredpalmer/kev-0.8b@v1.0` on a Colab GPU using 24 actual LANESHIFT schema-2
observations in `benchmark_requests.jsonl`.

It uses `kev.predictors.LocalPredictor` in-process. It does not start
`kev.serve`, expose an API, or measure batch throughput.

Run it on a T4/L4/A100 runtime with Python 3.12-3.13. Upload
`benchmark_requests.jsonl` when prompted. The notebook reports wall p50/p95,
model-time p50/p95, model load time, peak VRAM, input token counts, and the
measured CPU reference rows.

The fixture was generated from accepted `original_model` decisions in the
current recordings, rendered with the production scene prompt after the
L256 cap fix. Every request fits the graph budget; the maximum is 225 tokens.
