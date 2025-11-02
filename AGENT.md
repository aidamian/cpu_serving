# Agent Notes

## Objectives
- Benchmark Llama 3.1 8B inference on CPU across HuggingFace transformers, vLLM, and llama.cpp.
- Capture latency and memory metrics with reproducible scripts and a containerized environment.
- Document setup, execution, and result collection workflows.

## Completed
- [x] Created devcontainer with Python 3.11, CPU-only PyTorch, vLLM, llama-cpp-python, and build tooling.
- [x] Added reusable benchmarking utilities under `src/cpu_serving/`.
- [x] Implemented individual backend scripts (`scripts/benchmark_*.py`) with CLI options, JSON export, and memory tracking.
- [x] Implemented `scripts/run_all_benchmarks.py` for coordinated runs and consolidated reporting.
- [x] Updated `README.md` with setup instructions, run commands, and repository layout.

## Pending / Follow-up
- [ ] Download or convert GGUF weights for llama.cpp (place path in `--llamacpp-model-path`).
- [ ] Validate benchmarks on target hardware and capture baseline JSON reports in `artifacts/`.
- [ ] (Optional) Extend analysis notebooks or dashboards for automated comparison of multiple runs.
- [ ] (Optional) Integrate additional quantization strategies or smaller context windows for constrained systems.

## Usage Notes
- Launch the devcontainer before running scripts to ensure consistent dependencies.
- Authenticate with Hugging Face (`huggingface-cli login`) prior to first run.
- Supply `--print-json` to any script to stream raw metrics alongside the tabular view.
- Store generated artifacts under version control or external storage for longitudinal tracking.
