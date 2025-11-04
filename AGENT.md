# Agent Notes

## Objectives
- Benchmark Llama 3.1 8B inference on CPU across HuggingFace transformers, vLLM, and llama.cpp.
- Capture latency and memory metrics with reproducible scripts and a containerized environment.
- Document setup, execution, and result collection workflows.

## Completed
- [x] Created an Ubuntu 24.04 devcontainer with Python 3.12, `uv`, build tooling, and support for isolated virtual environments.
- [x] Added reusable benchmarking utilities under `src/cpu_serving/`.
- [x] Implemented individual backend scripts (`scripts/benchmark_*.py`) with CLI options, JSON export, and memory tracking.
- [x] Implemented `scripts/run_all_benchmarks.py` for coordinated runs, consolidated reporting, and automatic virtualenv provisioning.
- [x] Added `scripts/setup_virtualenvs.py` plus per-backend dependency manifests under `envs/`.
- [x] Updated `README.md` with setup instructions, run commands, and repository layout.

## Pending / Follow-up
- [ ] Download or convert GGUF weights for llama.cpp (place path in `--llamacpp-model-path`).
- [ ] Validate benchmarks on target hardware and capture baseline JSON reports in `artifacts/`.
- [ ] (Optional) Extend analysis notebooks or dashboards for automated comparison of multiple runs.
- [ ] (Optional) Integrate additional quantization strategies or smaller context windows for constrained systems.

## Usage Notes
- Launch the devcontainer before running scripts to ensure consistent dependencies.
- Run `python scripts/setup_virtualenvs.py` to provision venvs (skip if the orchestrator creates them for you).
- Authenticate with Hugging Face (`huggingface-cli login`) prior to first run.
- Supply `--print-json` to any script to stream raw metrics alongside the tabular view.
- Store generated artifacts under version control or external storage for longitudinal tracking.
