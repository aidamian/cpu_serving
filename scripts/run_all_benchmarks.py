#!/usr/bin/env python
"""Run all CPU benchmarks using isolated virtual environments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cpu_serving.benchmarks import short_model_name
from cpu_serving.console import log_color
from cpu_serving.venv_manager import (
    VirtualEnvError,
    available_backends,
    ensure_virtualenv,
    resolve_backend,
)

_DEFAULT_PROMPT = (
    "Write the DDL SQL for the definition of user accounts table. "
    "Output only the viable SQL."
)
_DEFAULT_MAX_NEW_TOKENS = 250
_DEFAULT_NUM_THREADS = 2
_DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
_DEFAULT_LLAMACPP_MODEL = (
    "./models/hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/"
    "llama-3.2-1b-instruct-q4_k_m.gguf"
)


_BACKEND_SCRIPTS: Dict[str, str] = {
    "huggingface": "benchmark_hf.py",
    "vllm": "benchmark_vllm.py",
    "llamacpp": "benchmark_llamacpp.py",
}


def _default_backends() -> List[str]:
    return list(available_backends())


def _iter_backends(selected: Sequence[str]) -> Iterable[str]:
    if not selected:
        yield from _default_backends()
        return
    for name in selected:
        yield resolve_backend(name)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |"
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = "\n".join(render(row) for row in rows)
    return "\n".join([header_line, separator, body])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU benchmarks for all configured backends using isolated environments."
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Prompt string to reuse for every backend.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a file containing the shared prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_MAX_NEW_TOKENS,
        help="Maximum completion tokens for each backend.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Subset of backends to run (hf, vllm, llamacpp). Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/simple_test"),
        help="Directory to store per-backend and summary JSON outputs.",
    )
    parser.add_argument(
        "--label",
        default="simple-test",
        help="Optional label to embed in output filenames.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Shared generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Shared top-p value.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Shared repetition penalty.",
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the final summary JSON to stdout.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not abort on backend failures; continue running remaining backends.",
    )

    parser.add_argument(
        "--skip-venv-sync",
        action="store_true",
        help="Reuse existing virtual environments without installing dependencies.",
    )
    parser.add_argument(
        "--venv-reinstall",
        action="store_true",
        help="Remove and recreate virtual environments before running.",
    )
    parser.add_argument(
        "--venv-upgrade",
        action="store_true",
        help="Upgrade dependencies to the latest allowed versions during sync.",
    )

    # HuggingFace options
    parser.add_argument(
        "--hf-model-id",
        default=_DEFAULT_MODEL_ID,
        help="Model repository for the HuggingFace backend.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional revision of the HuggingFace model.",
    )
    parser.add_argument(
        "--hf-tokenizer-id",
        default=None,
        help="Optional tokenizer repository override for HuggingFace.",
    )
    parser.add_argument(
        "--hf-dtype",
        default="float32",
        help="Torch dtype for HuggingFace.",
    )
    parser.add_argument(
        "--hf-num-threads",
        type=int,
        default=_DEFAULT_NUM_THREADS,
        help="CPU thread override for HuggingFace.",
    )
    parser.add_argument(
        "--hf-warmup",
        action="store_true",
        help="Enable warmup run for HuggingFace.",
    )
    parser.add_argument(
        "--hf-warmup-tokens",
        type=int,
        default=32,
        help="Warmup token count for HuggingFace.",
    )
    parser.add_argument(
        "--hf-quantize",
        action="append",
        choices=["int4", "int8"],
        default=None,
        help="Add bitsandbytes quantization runs for HuggingFace (int4/int8). Provide multiple times.",
    )
    parser.add_argument(
        "--hf-bitsandbytes-compute-dtype",
        default="float16",
        help="Compute dtype for bitsandbytes quantization (float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--hf-bitsandbytes-quant-type",
        default="nf4",
        help="Quantization type for 4-bit bitsandbytes runs (e.g. nf4).",
    )
    parser.add_argument(
        "--hf-bitsandbytes-disable-double-quant",
        action="store_true",
        help="Disable double quantization for 4-bit runs.",
    )
    parser.add_argument(
        "--hf-bitsandbytes-int8-cpu-offload",
        action="store_true",
        help="Enable FP32 CPU offload for int8 bitsandbytes runs.",
    )

    # vLLM options
    parser.add_argument(
        "--vllm-model-id",
        default=_DEFAULT_MODEL_ID,
        help="Model repository for vLLM.",
    )
    parser.add_argument(
        "--vllm-revision",
        default=None,
        help="Optional revision for the vLLM model.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="float32",
        help="Computation dtype for vLLM.",
    )
    parser.add_argument(
        "--vllm-num-threads",
        type=int,
        default=_DEFAULT_NUM_THREADS,
        help="CPU thread override for vLLM.",
    )
    parser.add_argument(
        "--vllm-warmup",
        action="store_true",
        help="Enable warmup run for vLLM.",
    )
    parser.add_argument(
        "--vllm-warmup-tokens",
        type=int,
        default=32,
        help="Warmup token count for vLLM.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (CPU runs should use 1).",
    )
    parser.add_argument(
        "--vllm-download-dir",
        default=None,
        help="Optional cache directory for vLLM model weights.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Force eager execution in vLLM.",
    )

    # llama.cpp options
    parser.add_argument(
        "--llamacpp-model-path",
        default=_DEFAULT_LLAMACPP_MODEL,
        help="Path to the llama.cpp GGUF model file.",
    )
    parser.add_argument(
        "--llamacpp-num-threads",
        type=int,
        default=_DEFAULT_NUM_THREADS,
        help="CPU thread override for llama.cpp.",
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
        help="Random seed for llama.cpp.",
    )
    parser.add_argument(
        "--llamacpp-warmup",
        action="store_true",
        help="Enable warmup run for llama.cpp.",
    )
    parser.add_argument(
        "--llamacpp-warmup-tokens",
        type=int,
        default=32,
        help="Warmup token count for llama.cpp.",
    )
    parser.add_argument(
        "--llamacpp-quantization",
        action="append",
        metavar="NAME=PATH",
        default=None,
        help="Additional quantized GGUF models for llama.cpp (forwarded to backend script).",
    )
    parser.add_argument(
        "--llamacpp-quantization-name",
        default=None,
        help="Label for the primary llama.cpp model path (forwarded).",
    )
    parser.add_argument(
        "--llamacpp-disable-auto-discover",
        action="store_true",
        help="Disable GGUF auto-discovery for llama.cpp benchmark.",
    )
    parser.add_argument(
        "--llamacpp-quantization-order",
        nargs="+",
        default=None,
        help="Ordering to report llama.cpp quantization labels.",
    )

    return parser.parse_args(argv)


def _build_common_args(args: argparse.Namespace) -> List[str]:
    cmd: List[str] = []
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.prompt_file:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--top-p", str(args.top_p)])
    cmd.extend(["--repetition-penalty", str(args.repetition_penalty)])
    return cmd


def _build_backend_command(
    backend: str,
    handle_python: Path,
    args: argparse.Namespace,
    output_path: Path,
) -> List[str]:
    script_dir = Path(__file__).resolve().parent
    script_name = _BACKEND_SCRIPTS[backend]
    script_path = script_dir / script_name

    cmd: List[str] = [str(handle_python), str(script_path), "--output", str(output_path)]
    cmd.extend(_build_common_args(args))

    if backend == "huggingface":
        cmd.extend(["--model-id", args.hf_model_id])
        if args.hf_revision:
            cmd.extend(["--revision", args.hf_revision])
        if args.hf_tokenizer_id:
            cmd.extend(["--tokenizer-id", args.hf_tokenizer_id])
        cmd.extend(["--dtype", args.hf_dtype])
        if args.hf_quantize:
            for mode in args.hf_quantize:
                cmd.extend(["--quantize", mode])
        cmd.extend(["--bitsandbytes-compute-dtype", args.hf_bitsandbytes_compute_dtype])
        cmd.extend(["--bitsandbytes-quant-type", args.hf_bitsandbytes_quant_type])
        if args.hf_bitsandbytes_disable_double_quant:
            cmd.append("--bitsandbytes-disable-double-quant")
        if args.hf_bitsandbytes_int8_cpu_offload:
            cmd.append("--bitsandbytes-int8-cpu-offload")
        if args.hf_num_threads is not None:
            cmd.extend(["--num-threads", str(args.hf_num_threads)])
        if args.hf_warmup:
            cmd.append("--warmup")
        cmd.extend(["--warmup-tokens", str(args.hf_warmup_tokens)])
    elif backend == "vllm":
        cmd.extend(["--model-id", args.vllm_model_id])
        if args.vllm_revision:
            cmd.extend(["--revision", args.vllm_revision])
        cmd.extend(["--dtype", args.vllm_dtype])
        cmd.extend(["--tensor-parallel-size", str(args.vllm_tensor_parallel_size)])
        if args.vllm_download_dir:
            cmd.extend(["--download-dir", str(args.vllm_download_dir)])
        if args.vllm_enforce_eager:
            cmd.append("--enforce-eager")
        if args.vllm_num_threads is not None:
            cmd.extend(["--num-threads", str(args.vllm_num_threads)])
        if args.vllm_warmup:
            cmd.append("--warmup")
        cmd.extend(["--warmup-tokens", str(args.vllm_warmup_tokens)])
    elif backend == "llamacpp":
        if not args.llamacpp_model_path:
            raise VirtualEnvError(
                "llama.cpp backend selected but --llamacpp-model-path was not provided."
            )
        cmd.extend(["--model-path", str(args.llamacpp_model_path)])
        if args.llamacpp_quantization_name:
            cmd.extend(["--quantization-name", args.llamacpp_quantization_name])
        if args.llamacpp_quantization:
            for item in args.llamacpp_quantization:
                cmd.extend(["--quantization", item])
        if args.llamacpp_disable_auto_discover:
            cmd.append("--disable-auto-discover")
        if args.llamacpp_quantization_order:
            cmd.extend(["--quantization-order", *args.llamacpp_quantization_order])
        if args.llamacpp_num_threads is not None:
            cmd.extend(["--num-threads", str(args.llamacpp_num_threads)])
        cmd.extend(["--n-ctx", str(args.llamacpp_n_ctx)])
        cmd.extend(["--n-batch", str(args.llamacpp_n_batch)])
        cmd.extend(["--seed", str(args.llamacpp_seed)])
        if args.llamacpp_warmup:
            cmd.append("--warmup")
        cmd.extend(["--warmup-tokens", str(args.llamacpp_warmup_tokens)])
    else:
        raise VirtualEnvError(f"Unsupported backend '{backend}'.")

    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backends = list(dict.fromkeys(_iter_backends(args.backends)))
    if not backends:
        print("No backends selected.", file=sys.stderr)
        return 1

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label_suffix = f"_{args.label}" if args.label else ""

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[List[str]] = []
    combined_results: List[Dict[str, object]] = []
    backend_payloads: Dict[str, Dict[str, object]] = {}
    failures: List[Dict[str, object]] = []

    for backend in backends:
        log_color(f"\n=== Running backend: {backend} ===", "b")
        try:
            handle = ensure_virtualenv(
                backend,
                sync_dependencies=not args.skip_venv_sync,
                reinstall=args.venv_reinstall,
                upgrade=args.venv_upgrade,
            )
        except VirtualEnvError as exc:
            print(f"Failed to prepare virtualenv for {backend}: {exc}", file=sys.stderr)
            return 2

        backend_dir = args.output_dir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        output_path = backend_dir / f"{timestamp}{label_suffix}.json"

        cmd = _build_backend_command(backend, handle.python, args, output_path)
        log_color(f"Executing: {' '.join(cmd)}", "d")
        try:
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)
        except subprocess.CalledProcessError as exc:
            message = f"Backend '{backend}' failed with exit code {exc.returncode}."
            print(message, file=sys.stderr)
            if args.continue_on_error:
                failures.append(
                    {
                        "backend": backend,
                        "return_code": exc.returncode,
                        "command": cmd,
                    }
                )
                continue
            return exc.returncode or 3

        if not output_path.exists():
            print(f"Expected output file {output_path} was not created.", file=sys.stderr)
            if args.continue_on_error:
                failures.append(
                    {
                        "backend": backend,
                        "return_code": 4,
                        "command": cmd,
                        "error": "missing_output",
                    }
                )
                continue
            return 4

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        backend_payloads[backend] = payload
        for result in payload.get("results", []):
            combined_results.append(result)
            parameters = result.get("parameters") or {}
            quant_label = parameters.get("quantization") or parameters.get("dtype") or "-"
            quant_detail = parameters.get("quantization_detail")
            if quant_label != "-" and quant_detail:
                quant_label = f"{quant_label} ({quant_detail})"
            summary_rows.append(
                [
                    backend,
                    short_model_name(result.get("model") or ""),
                    str(quant_label),
                    str(result.get("num_threads") or "-"),
                    f"{float(result.get('load_time_s', 0.0)):.2f}",
                    f"{float(result.get('generate_time_s', 0.0)):.2f}",
                    str(result.get("completion_tokens", "")),
                    f"{float(result.get('tokens_per_second', 0.0)):.2f}",
                    f"{float(result.get('peak_memory_bytes', 0.0)) / (1024 ** 2):.1f}",
                ]
            )

    if combined_results:
        headers = [
            "Backend",
            "Model",
            "Quantization",
            "Threads",
            "Load (s)",
            "Gen (s)",
            "Tokens",
            "Tok/s",
            "Peak Mem (MiB)",
        ]
        log_color("\n=== Summary ===", "b")
        log_color(_format_table(headers, summary_rows), "g")

    summary_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "label": args.label,
        "results": combined_results,
        "per_backend": backend_payloads,
        "failures": failures,
    }

    summary_path = args.output_dir / f"summary_{timestamp}{label_suffix}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log_color(f"\nSaved combined results to {summary_path}", "g")

    if args.print_json:
        log_color(json.dumps(summary_payload, indent=2), "d")

    if failures:
        print(
            f"{len(failures)} backend(s) failed; see summary for details.",
            file=sys.stderr,
        )
    if failures and not args.continue_on_error:
        first = failures[0]
        return int(first.get("return_code") or 3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
