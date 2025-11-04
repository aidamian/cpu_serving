"""
Core benchmarking utilities for CPU-based Llama 3.1 8B evaluations.
"""

from .benchmarks import (
    HFBenchmarkConfig,
    VLLMBenchmarkConfig,
    LlamaCppBenchmarkConfig,
    BenchmarkResult,
    run_hf_benchmark,
    run_vllm_benchmark,
    run_llamacpp_benchmark,
    aggregate_results,
)
from .venv_manager import (
    VirtualEnvError,
    available_backends,
    ensure_virtualenv,
)

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
