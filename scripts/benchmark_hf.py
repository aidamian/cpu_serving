#!/usr/bin/env python
"""Benchmark HuggingFace transformers (PyTorch) on CPU for Llama 3.1 8B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from cpu_serving.benchmarks import (
    HFBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_hf_benchmark,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-memory and latency benchmark for HuggingFace transformers."
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model repository to load.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional specific revision/commit/tag of the model repository.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Optional tokenizer repository to override the model default.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype to use (float32, bfloat16, float16). Defaults to float32.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt string to feed the model. Defaults to the shared benchmark prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the prompt to benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Override the number of CPU threads used by PyTorch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Defaults to deterministic output.",
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
        help="Run an unmeasured warmup generation before benchmarking.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="Maximum new tokens for the optional warmup pass.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store the JSON benchmark result.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the raw JSON results payload to stdout.",
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

    default_prompt = HFBenchmarkConfig().prompt

    config = HFBenchmarkConfig(
        model_id=args.model_id,
        revision=args.revision,
        tokenizer_id=args.tokenizer_id,
        dtype=args.dtype,
        prompt=prompt or default_prompt,
        max_new_tokens=args.max_new_tokens,
        num_threads=args.num_threads,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_warmup=args.warmup,
        warmup_tokens=args.warmup_tokens,
    )

    result = run_hf_benchmark(config)

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
