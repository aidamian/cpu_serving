# CPU Serving Benchmarks

Compare memory footprint and latency of running **Llama 3.1 8B** on CPU across three inference stacks:

- `huggingface` / PyTorch transformers
- `vllm`
- `llama.cpp` (GGUF weights)

The repository provides a ready-to-use devcontainer, standalone backend scripts, and an orchestrated runner that captures structured results for analysis.

## Quickstart

- Install VS Code with the Dev Containers extension (or the `devcontainer` CLI) and accept access to the `meta-llama/Llama-3.1-8B` model on Hugging Face.
- Clone the repo and open it in the devcontainer:
  ```bash
  devcontainer open .
  ```
  The container installs Python 3.11, CPU-only PyTorch, vLLM, llama.cpp bindings, and supporting tools via `requirements.txt`.
- Authenticate with Hugging Face inside the container so downloads succeed:
  ```bash
  huggingface-cli login
  ```

## Model Assets

- **Transformers & vLLM** load weights directly from Hugging Face. Set `HF_HOME` or `--download-dir` if you need a custom cache location. CPU runs require ~16–24 GB of RAM for the full-precision model; consider parameter-efficient or quantized variants if memory is constrained.
- **llama.cpp** requires a GGUF file. Either download an official GGUF release (e.g. `meta-llama/Llama-3.1-8B-Instruct-GGUF`) or convert `meta-llama/Llama-3.1-8B` locally using the conversion tools in the upstream `llama.cpp` repository. Place the resulting file under `models/` (or supply an absolute path when running benchmarks).

## Single Backend Benchmarks

Each script reports peak resident memory, load time, generation latency, and tokens-per-second. Use `--help` on any script for options.

- PyTorch / Transformers:
  ```bash
  python scripts/benchmark_hf.py \
    --model-id meta-llama/Llama-3.1-8B \
    --max-new-tokens 128 \
    --num-threads 16
  ```
- vLLM (CPU eager mode recommended):
  ```bash
  python scripts/benchmark_vllm.py \
    --model-id meta-llama/Llama-3.1-8B \
    --enforce-eager \
    --num-threads 16
  ```
- llama.cpp (GGUF input required):
  ```bash
  python scripts/benchmark_llamacpp.py \
    --model-path ./models/llama-3.1-8b-q4_k_m.gguf \
    --num-threads 16
  ```

All scripts accept `--prompt` or `--prompt-file` for custom inputs and can emit raw JSON via `--print-json`.

## Full Comparative Run

`python scripts/run_all_benchmarks.py` orchestrates every backend, collates results, and saves a timestamped report under `artifacts/`.

Example:
```bash
python scripts/run_all_benchmarks.py \
  --llamacpp-model-path ./models/llama-3.1-8b-q4_k_m.gguf \
  --max-new-tokens 128 \
  --hf-num-threads 16 \
  --vllm-num-threads 16 \
  --llamacpp-num-threads 16 \
  --label local-test
```

Use `--backends` to run a subset (e.g. `--backends hf vllm`) and `--print-json` to mirror the saved payload to stdout. Any backend failures are captured in the JSON under `errors`.

## Results

The aggregated JSON includes:
- System metadata (CPU topology, RAM, Python version).
- Per-backend metrics (`load_time_s`, `generate_time_s`, `completion_tokens`, `tokens_per_second`, `peak_memory_mebibytes`).
- The generated completion text for quick sanity checks.

Saved files follow `artifacts/benchmark_<timestamp>[_label].json`. They can be post-processed with pandas or imported into dashboards.

## Repository Layout

- `.devcontainer/` – devcontainer configuration, Dockerfile, and post-create installer.
- `requirements.txt` – Python dependencies (CPU-only PyTorch, vLLM, llama-cpp-python, etc.).
- `scripts/` – entry points for individual backends and the orchestration runner.
- `src/cpu_serving/` – reusable benchmarking utilities (memory sampling, result formatting).
- `artifacts/` – output directory for JSON reports (kept empty via `.gitkeep`).

## Notes & Tips

- Large CPU runs benefit from setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `VLLM_WORKER_CPU_THREADS`; the scripts set these automatically when `--num-threads` is provided.
- vLLM CPU support is evolving; enabling `--enforce-eager` often yields more predictable behavior at the expense of throughput.
- Peak memory estimates rely on periodic RSS sampling through `psutil`; adjust the sampling rate in `src/cpu_serving/benchmarks.py` if more precision is required.
