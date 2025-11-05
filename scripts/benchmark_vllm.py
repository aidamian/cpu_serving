#!/usr/bin/env python
"""Benchmark vLLM CPU execution for Llama 3.2 1B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from cpu_serving.benchmarks import (
    VLLMBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_vllm_benchmark,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU benchmark for the vLLM inference backend."
    )
    defaults = VLLMBenchmarkConfig()
    parser.add_argument(
        "--model-id",
        default=defaults.model_id,
        help="HuggingFace model repository to evaluate.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/commit hash for the model repository.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Computation dtype to use (float32, bfloat16, float16).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=defaults.tensor_parallel_size,
        help="Tensor parallelism degree. For CPU runs, keep this at 1.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional custom cache directory for model weights.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager mode in vLLM (recommended for CPU).",
    )
    parser.add_argument(
        "--prompt",
        default=defaults.prompt,
        help="Prompt string to feed the model.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults.max_new_tokens,
        help="Maximum number of completion tokens.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=defaults.num_threads,
        help="Override the number of CPU threads vLLM should use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty during generation.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run an un-timed warmup generation before benchmarking.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="Maximum tokens for the warmup pass.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON output.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON payload to stdout.",
    )
    return parser


def _load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> Optional[str]:
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        return text.strip()
    return prompt


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompt = _load_prompt(args.prompt, args.prompt_file)
    default_prompt = VLLMBenchmarkConfig().prompt

    config = VLLMBenchmarkConfig(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
        enforce_eager=args.enforce_eager,
        prompt=prompt or default_prompt,
        max_new_tokens=args.max_new_tokens,
        num_threads=args.num_threads,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_warmup=args.warmup,
        warmup_tokens=args.warmup_tokens,
    )

    result = run_vllm_benchmark(config)
    print(format_results_table([result]))
    print()
    print("Completion preview:")
    print(result.completion)

    payload = aggregate_results([result])
    if args.output:
        write_results(payload, args.output)
        print(f"\nSaved results to {args.output}")

    if args.print_json:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
