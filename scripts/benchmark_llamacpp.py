#!/usr/bin/env python
"""Benchmark llama.cpp CPU execution for Llama 3.2 1B GGUF weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from cpu_serving.benchmarks import (
    LlamaCppBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_llamacpp_quantized_benchmarks,
    write_results,
)
from cpu_serving.console import log_color


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU benchmark for llama.cpp GGUF weights."
    )
    defaults = LlamaCppBenchmarkConfig()
    parser.add_argument(
        "--model-path",
        default=defaults.model_path,
        help="Path to the GGUF model file (e.g. Llama 3.2 1B quantized weights).",
    )
    parser.add_argument(
        "--quantization-name",
        default=None,
        help="Optional label for the primary --model-path quantization (e.g. int4).",
    )
    parser.add_argument(
        "--quantization",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Additional quantized GGUF model to benchmark, formatted as NAME=PATH. "
            "Provide multiple times to benchmark several quantizations."
        ),
    )
    parser.add_argument(
        "--disable-auto-discover",
        action="store_true",
        help="Disable automatic discovery of sibling GGUF files alongside --model-path.",
    )
    parser.add_argument(
        "--quantization-order",
        nargs="+",
        default=None,
        help="Optional ordering for reported quantization labels (e.g. int4 int8).",
    )
    parser.add_argument(
        "--prompt",
        default=defaults.prompt,
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
        default=defaults.max_new_tokens,
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
        default=defaults.num_threads,
        help="Override number of CPU threads. Defaults to 2.",
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

    quantizations: Dict[str, str] = {}
    if args.quantization:
        for entry in args.quantization:
            if "=" not in entry:
                parser.error(f"Invalid --quantization value '{entry}'. Expected NAME=PATH.")
            name, path = entry.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name or not path:
                parser.error(f"Invalid --quantization value '{entry}'. Expected NAME=PATH.")
            quantizations[name] = path

    config = LlamaCppBenchmarkConfig(
        model_path=args.model_path,
        quantization_name=args.quantization_name,
        quantizations=quantizations,
        auto_discover_quantizations=not args.disable_auto_discover,
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

    results = run_llamacpp_quantized_benchmarks(config, quantization_order=args.quantization_order)
    log_color(format_results_table(results), "b")
    print("", flush=True)
    for result in results:
        label = result.parameters.get("quantization") or Path(result.model).name
        log_color(f"[{label}] Completion preview:", "y")
        log_color(result.completion, "g")
        print("", flush=True)

    payload = aggregate_results(results)
    if args.output:
        write_results(payload, args.output)
        print("", flush=True)
        log_color(f"Saved results to {args.output}", "g")

    if args.print_json:
        log_color(json.dumps(payload, indent=2), "d")


if __name__ == "__main__":
    main()
