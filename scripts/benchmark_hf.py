#!/usr/bin/env python
"""Benchmark HuggingFace transformers (PyTorch) on CPU for Llama 3.2 1B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from cpu_serving.benchmarks import (
    HFBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_hf_benchmark,
    run_hf_quantized_benchmarks,
    write_results,
)
from cpu_serving.console import log_color


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-memory and latency benchmark for HuggingFace transformers."
    )
    defaults = HFBenchmarkConfig()
    parser.add_argument(
        "--model-id",
        default=defaults.model_id,
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
        "--quantize",
        dest="quantizations",
        action="append",
        choices=["int4", "int8"],
        help="Add a bitsandbytes quantization run (int4 or int8). Provide multiple times to benchmark several modes.",
    )
    parser.add_argument(
        "--bitsandbytes-compute-dtype",
        default=defaults.bitsandbytes_compute_dtype,
        help="Compute dtype for bitsandbytes quantization (float16, float32, bfloat16).",
    )
    parser.add_argument(
        "--bitsandbytes-quant-type",
        default=defaults.bitsandbytes_quant_type,
        help="4-bit quantization type for bitsandbytes (e.g. nf4, fp4).",
    )
    parser.add_argument(
        "--bitsandbytes-disable-double-quant",
        action="store_true",
        help="Disable double quantization when running 4-bit benchmarks.",
    )
    parser.add_argument(
        "--bitsandbytes-int8-cpu-offload",
        action="store_true",
        help="Enable FP32 CPU offload for int8 quantization runs.",
    )
    parser.add_argument(
        "--prompt",
        default=defaults.prompt,
        help="Prompt string to feed the model.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the prompt to benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults.max_new_tokens,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=defaults.num_threads,
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
        bitsandbytes_compute_dtype=args.bitsandbytes_compute_dtype,
        bitsandbytes_quant_type=args.bitsandbytes_quant_type,
        bitsandbytes_use_double_quant=not args.bitsandbytes_disable_double_quant,
        bitsandbytes_int8_cpu_offload=args.bitsandbytes_int8_cpu_offload,
        prompt=prompt or default_prompt,
        max_new_tokens=args.max_new_tokens,
        num_threads=args.num_threads,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_warmup=args.warmup,
        warmup_tokens=args.warmup_tokens,
    )

    quantization_modes: List[str]
    if args.quantizations is None:
        quantization_modes = ["int4", "int8"]
    else:
        quantization_modes = args.quantizations

    results = [run_hf_benchmark(config)]
    if quantization_modes:
        quantized_results = run_hf_quantized_benchmarks(config, quantizations=quantization_modes)
        results.extend(quantized_results)

    log_color(format_results_table(results), "b")
    print("", flush=True)
    for result in results:
        label = result.parameters.get("quantization") or result.parameters.get("dtype") or "default"
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
