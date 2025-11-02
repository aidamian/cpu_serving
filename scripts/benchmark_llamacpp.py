#!/usr/bin/env python
"""Benchmark llama.cpp CPU execution for Llama 3.1 8B GGUF weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from cpu_serving.benchmarks import (
    LlamaCppBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_llamacpp_benchmark,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU benchmark for llama.cpp GGUF weights."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the GGUF model file (e.g. Llama 3.1 8B quantized weights).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt string for the benchmark run.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a file containing the prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context window size for llama.cpp.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="Batch size for prompt ingestion in llama.cpp.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Override number of CPU threads. Defaults to all logical cores.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by llama.cpp.",
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
        help="Run an un-timed warmup call before measuring.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="Max tokens for the warmup call.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
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
    default_prompt = LlamaCppBenchmarkConfig().prompt

    config = LlamaCppBenchmarkConfig(
        model_path=args.model_path,
        prompt=prompt or default_prompt,
        max_new_tokens=args.max_new_tokens,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        num_threads=args.num_threads,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_warmup=args.warmup,
        warmup_tokens=args.warmup_tokens,
    )

    result = run_llamacpp_benchmark(config)
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
