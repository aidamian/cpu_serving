"""Core benchmarking utilities for CPU-based Llama evaluations."""

from __future__ import annotations

import importlib
from typing import Any, Dict

__all__ = [
    "HFBenchmarkConfig",
    "VLLMBenchmarkConfig",
    "LlamaCppBenchmarkConfig",
    "BenchmarkResult",
    "run_hf_benchmark",
    "run_vllm_benchmark",
    "run_llamacpp_benchmark",
    "aggregate_results",
    "available_backends",
    "ensure_virtualenv",
    "VirtualEnvError",
]

_BENCHMARK_EXPORTS = {
    "HFBenchmarkConfig",
    "VLLMBenchmarkConfig",
    "LlamaCppBenchmarkConfig",
    "BenchmarkResult",
    "run_hf_benchmark",
    "run_vllm_benchmark",
    "run_llamacpp_benchmark",
    "aggregate_results",
}

_VENV_EXPORTS = {
    "available_backends",
    "ensure_virtualenv",
    "VirtualEnvError",
}


def __getattr__(name: str) -> Any:
    if name in _BENCHMARK_EXPORTS:
        module = importlib.import_module(".benchmarks", __name__)
        return getattr(module, name)
    if name in _VENV_EXPORTS:
        module = importlib.import_module(".venv_manager", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
