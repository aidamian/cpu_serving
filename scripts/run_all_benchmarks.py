#!/usr/bin/env python
"""Run the full CPU benchmark suite across HuggingFace, vLLM, and llama.cpp backends."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cpu_serving.benchmarks import (
    HFBenchmarkConfig,
    LlamaCppBenchmarkConfig,
    VLLMBenchmarkConfig,
    aggregate_results,
    format_results_table,
    run_hf_benchmark,
    run_llamacpp_benchmark,
    run_vllm_benchmark,
    write_results,
)


BACKEND_ALIASES = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "vllm": "vllm",
    "llamacpp": "llama.cpp",
    "llama.cpp": "llama.cpp",
}

ALL_BACKENDS = ("huggingface", "vllm", "llama.cpp")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU benchmarks across multiple Llama 3.1 8B inference stacks."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt string used for every backend.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional file containing the benchmark prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max completion tokens for all backends (can be overridden individually).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(ALL_BACKENDS),
        help="Subset of backends to run: hf, vllm, llamacpp. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where benchmark JSON reports will be stored.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label appended to the output filename for easier tracking.",
    )

    # HuggingFace specific options
    parser.add_argument(
        "--hf-model-id",
        default="meta-llama/Llama-3.1-8B",
        help="Model repository for the HuggingFace backend.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional revision of the HuggingFace model.",
    )
    parser.add_argument(
        "--hf-dtype",
        default="float32",
        help="Torch dtype for HuggingFace backend.",
    )
    parser.add_argument(
        "--hf-num-threads",
        type=int,
        default=None,
        help="Thread count override for HuggingFace backend.",
    )
    parser.add_argument(
        "--hf-warmup",
        action="store_true",
        help="Enable warmup pass for HuggingFace backend.",
    )

    # vLLM specific options
    parser.add_argument(
        "--vllm-model-id",
        default="meta-llama/Llama-3.1-8B",
        help="Model repository for vLLM backend.",
    )
    parser.add_argument(
        "--vllm-revision",
        default=None,
        help="Optional revision of the vLLM model.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="float32",
        help="Computation dtype for vLLM.",
    )
    parser.add_argument(
        "--vllm-num-threads",
        type=int,
        default=None,
        help="Thread count override for vLLM backend.",
    )
    parser.add_argument(
        "--vllm-warmup",
        action="store_true",
        help="Enable warmup pass for vLLM backend.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Force eager execution mode in vLLM (recommended for CPU).",
    )

    # llama.cpp specific options
    parser.add_argument(
        "--llamacpp-model-path",
        default=None,
        help="Path to the llama.cpp GGUF weights. Required when running the llama.cpp backend.",
    )
    parser.add_argument(
        "--llamacpp-num-threads",
        type=int,
        default=None,
        help="Thread count override for llama.cpp backend.",
    )
    parser.add_argument(
        "--llamacpp-n-ctx",
        type=int,
        default=4096,
        help="Context window for llama.cpp.",
    )
    parser.add_argument(
        "--llamacpp-n-batch",
        type=int,
        default=512,
        help="Prompt ingestion batch size for llama.cpp.",
    )
    parser.add_argument(
        "--llamacpp-seed",
        type=int,
        default=42,
        help="Random seed for llama.cpp backend.",
    )
    parser.add_argument(
        "--llamacpp-warmup",
        action="store_true",
        help="Enable warmup pass for llama.cpp backend.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Shared temperature parameter (applies to all backends unless overridden).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Shared top-p parameter.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Shared repetition penalty parameter.",
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the combined JSON report to stdout.",
    )

    return parser.parse_args(argv)


def _normalize_backends(requested: Sequence[str]) -> List[str]:
    normalized = []
    for item in requested:
        lower = item.lower()
        if lower not in BACKEND_ALIASES:
            raise ValueError(f"Unknown backend '{item}'. Valid options: hf, vllm, llamacpp.")
        canonical = BACKEND_ALIASES[lower]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> Optional[str]:
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        return text.strip()
    return prompt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        backends = _normalize_backends(args.backends)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    prompt = _load_prompt(args.prompt, args.prompt_file)
    default_prompt = HFBenchmarkConfig().prompt
    resolved_prompt = prompt or default_prompt

    results = []
    errors: Dict[str, str] = {}

    if "huggingface" in backends:
        try:
            hf_config = HFBenchmarkConfig(
                model_id=args.hf_model_id,
                revision=args.hf_revision,
                dtype=args.hf_dtype,
                prompt=resolved_prompt,
                max_new_tokens=args.max_new_tokens,
                num_threads=args.hf_num_threads,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_warmup=args.hf_warmup,
            )
            results.append(run_hf_benchmark(hf_config))
        except Exception as exc:  # pylint: disable=broad-except
            errors["huggingface"] = "".join(
                traceback.format_exception_only(exc.__class__, exc)
            ).strip()

    if "vllm" in backends:
        try:
            vllm_config = VLLMBenchmarkConfig(
                model_id=args.vllm_model_id,
                revision=args.vllm_revision,
                dtype=args.vllm_dtype,
                prompt=resolved_prompt,
                max_new_tokens=args.max_new_tokens,
                num_threads=args.vllm_num_threads,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_warmup=args.vllm_warmup,
                enforce_eager=args.vllm_enforce_eager,
            )
            results.append(run_vllm_benchmark(vllm_config))
        except Exception as exc:  # pylint: disable=broad-except
            errors["vllm"] = "".join(
                traceback.format_exception_only(exc.__class__, exc)
            ).strip()

    if "llama.cpp" in backends:
        model_path = args.llamacpp_model_path
        if not model_path:
            errors["llama.cpp"] = "Missing --llamacpp-model-path argument."
        else:
            path = Path(model_path)
            if not path.exists():
                errors["llama.cpp"] = f"GGUF model not found at {path}"
            else:
                try:
                    llamacpp_config = LlamaCppBenchmarkConfig(
                        model_path=str(path),
                        prompt=resolved_prompt,
                        max_new_tokens=args.max_new_tokens,
                        num_threads=args.llamacpp_num_threads,
                        n_ctx=args.llamacpp_n_ctx,
                        n_batch=args.llamacpp_n_batch,
                        seed=args.llamacpp_seed,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        do_warmup=args.llamacpp_warmup,
                    )
                    results.append(run_llamacpp_benchmark(llamacpp_config))
                except Exception as exc:  # pylint: disable=broad-except
                    errors["llama.cpp"] = "".join(
                        traceback.format_exception_only(exc.__class__, exc)
                    ).strip()

    if results:
        print(format_results_table(results))
    else:
        print("No successful benchmark results to display.", file=sys.stderr)

    if errors:
        print("\nErrors:", file=sys.stderr)
        for backend, message in errors.items():
            print(f"- {backend}: {message}", file=sys.stderr)

    payload = aggregate_results(results)
    if errors:
        payload["errors"] = errors

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.label}" if args.label else ""
    output_path = args.output_dir / f"benchmark_{timestamp}{suffix}.json"
    write_results(payload, output_path)
    print(f"\nSaved combined results to {output_path}")

    if args.print_json:
        print(json.dumps(payload, indent=2))

    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
