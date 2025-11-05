from __future__ import annotations

import errno
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import psutil


DEFAULT_PROMPT = "Quick CPU smoke-test prompt for the Hugging Face backend."


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return "float32"
    if normalized in {"bfloat16", "bf16"}:
        return "bfloat16"
    if normalized in {"float16", "fp16"}:
        return "float16"
    raise ValueError(f"Unsupported dtype '{dtype}'")


class MemoryMonitor:
    """Samples process memory usage in a background thread."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._process = psutil.Process(os.getpid())
        self.max_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "MemoryMonitor":
        self.max_rss_bytes = self._process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self.max_rss_bytes:
                    self.max_rss_bytes = rss
            except psutil.Error:
                # Process might have ended; bail out.
                break
            time.sleep(self.interval_s)


@dataclass
class BenchmarkResult:
    backend: str
    model: str
    prompt: str
    prompt_tokens: int
    completion: str
    completion_tokens: int
    max_new_tokens: int
    load_time_s: float
    generate_time_s: float
    peak_memory_bytes: int
    num_threads: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.generate_time_s == 0:
            return float("inf")
        return self.completion_tokens / self.generate_time_s

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        payload["tokens_per_second"] = self.tokens_per_second
        payload["peak_memory_mebibytes"] = self.peak_memory_bytes / (1024 ** 2)
        return payload


@dataclass
class BaseBenchmarkConfig:
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = 8
    num_threads: Optional[int] = 2
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_warmup: bool = False
    warmup_tokens: int = 16


@dataclass
class HFBenchmarkConfig(BaseBenchmarkConfig):
    model_id: str = "meta-llama/Llama-3.2-1B"
    revision: Optional[str] = None
    dtype: str = "float32"
    trust_remote_code: bool = False
    tokenizer_id: Optional[str] = None


@dataclass
class VLLMBenchmarkConfig(BaseBenchmarkConfig):
    model_id: str = "meta-llama/Llama-3.2-1B"
    revision: Optional[str] = None
    dtype: str = "float32"
    tensor_parallel_size: int = 1
    download_dir: Optional[str] = None
    enforce_eager: bool = False


@dataclass
class LlamaCppBenchmarkConfig(BaseBenchmarkConfig):
    model_path: str = (
        "./models/hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/"
        "llama-3.2-1b-instruct-q4_k_m.gguf"
    )
    n_ctx: int = 4096
    n_batch: int = 512
    seed: int = 42


_VLLM_IPC_PATCHED = False


def _ensure_vllm_ipc_support() -> None:
    """Force vLLM to fall back to TCP sockets if IPC is not permitted."""

    global _VLLM_IPC_PATCHED
    if _VLLM_IPC_PATCHED:
        return

    try:
        import zmq  # type: ignore
        from vllm import utils as vllm_utils  # type: ignore
    except ImportError:
        return

    ctx = zmq.Context()  # type: ignore[attr-defined]
    socket_path = Path(tempfile.gettempdir()) / f"cpu-serving-zmq-{uuid4().hex}"
    ipc_uri = f"ipc://{socket_path}"
    original_get_ipc = getattr(vllm_utils, "get_open_zmq_ipc_path", None)
    sock = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
    try:
        sock.bind(ipc_uri)
    except zmq.ZMQError as exc:  # type: ignore[attr-defined]
        if exc.errno not in {getattr(zmq, "EPERM", errno.EPERM), errno.EPERM}:
            return

        def _tcp_uri() -> str:
            try:
                port = vllm_utils.get_open_port()
                return f"tcp://127.0.0.1:{port}"
            except OSError:
                if callable(original_get_ipc):
                    return original_get_ipc()
                return ipc_uri

        vllm_utils.get_open_zmq_ipc_path = _tcp_uri  # type: ignore[assignment]
        _VLLM_IPC_PATCHED = True
    finally:
        try:
            sock.close(0)
        except Exception:
            pass
        ctx.term()
        try:
            socket_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_threads_config(num_threads: Optional[int]) -> None:
    if num_threads is None or num_threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["VLLM_WORKER_CPU_THREADS"] = str(num_threads)
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except ImportError:
        pass


def _run_warmup(generate_fn, warmup_tokens: int) -> None:
    try:
        generate_fn(warmup_tokens)
    except Exception:
        # Warmup is best-effort; ignore failures so the main run can proceed.
        return


def run_hf_benchmark(config: HFBenchmarkConfig) -> BenchmarkResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_threads_config(config.num_threads)
    dtype = _normalize_dtype(config.dtype)
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    model_id = config.model_id
    tokenizer_id = config.tokenizer_id or model_id

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=config.revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=config.revision,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=config.trust_remote_code,
    )
    load_time = time.perf_counter() - load_start

    if config.do_warmup:
        warmup_inputs = tokenizer(
            "Warmup prompt.", return_tensors="pt"
        )
        warmup_inputs = {k: v.to("cpu") for k, v in warmup_inputs.items()}
        with torch.inference_mode():
            model.generate(
                **warmup_inputs,
                max_new_tokens=config.warmup_tokens,
                do_sample=False,
            )

    inputs = tokenizer(config.prompt, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].shape[-1]
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with MemoryMonitor() as monitor:
        generate_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.temperature > 0,
            )
        generate_time = time.perf_counter() - generate_start

    generated_tokens = output_ids.shape[-1] - prompt_tokens
    completion = tokenizer.decode(
        output_ids[0][prompt_tokens:], skip_special_tokens=True
    )

    return BenchmarkResult(
        backend="huggingface",
        model=model_id,
        prompt=config.prompt,
        prompt_tokens=prompt_tokens,
        completion=completion.strip(),
        completion_tokens=generated_tokens,
        max_new_tokens=config.max_new_tokens,
        load_time_s=load_time,
        generate_time_s=generate_time,
        peak_memory_bytes=monitor.max_rss_bytes,
        num_threads=config.num_threads,
        parameters={
            "dtype": dtype,
            "revision": config.revision,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
    )


_VLLM_CPU_PATCHED = False


def run_vllm_benchmark(config: VLLMBenchmarkConfig) -> BenchmarkResult:
    from vllm import LLM, SamplingParams
    from vllm import platforms as vllm_platforms
    try:
        from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "vLLM CPU platform is unavailable; install a CPU-enabled build."
        ) from exc

    _ensure_threads_config(config.num_threads)
    dtype = _normalize_dtype(config.dtype)

    current_platform = vllm_platforms.current_platform
    if not current_platform.is_cpu():
        current_platform.__class__ = CpuPlatform
        current_platform.device_type = "cpu"
        current_platform.device_name = "cpu"
        current_platform.dispatch_key = "CPU"
        current_platform.dist_backend = "gloo"
        current_platform._enum = CpuPlatform._enum

    global _VLLM_CPU_PATCHED
    if not _VLLM_CPU_PATCHED:
        original_method = CpuPlatform.get_allowed_cpu_core_node_list

        def _patched_get_allowed_cpu_core_node_list(cls):  # type: ignore[override]
            try:
                return original_method.__func__(cls)  # type: ignore[attr-defined]
            except (
                json.JSONDecodeError,
                FileNotFoundError,
                subprocess.CalledProcessError,
            ):
                try:
                    lscpu_output = subprocess.check_output(
                        "lscpu -J -e=CPU,CORE,NODE", shell=True, text=True
                    )
                except Exception:
                    return original_method.__func__(cls)  # type: ignore[attr-defined]

                sanitized = lscpu_output.replace('"node": -', '"node": 0')
                decoded = json.loads(
                    sanitized, object_hook=LogicalCPUInfo.json_decoder
                )
                logical_cpu_list: List[LogicalCPUInfo] = decoded.get("cpus", [])

                allowed_cpu_ids = sorted(os.sched_getaffinity(0))
                logical_cpu_list = [
                    cpu for cpu in logical_cpu_list if cpu.id in allowed_cpu_ids
                ]

                if not logical_cpu_list:
                    logical_cpu_list = [
                        LogicalCPUInfo(
                            id=cpu_id,
                            physical_core=cpu_id,
                            numa_node=0,
                        )
                        for cpu_id in allowed_cpu_ids
                    ]

                for cpu in logical_cpu_list:
                    if getattr(cpu, "numa_node", 0) < 0:
                        cpu.numa_node = 0  # type: ignore[attr-defined]

                allowed_numa_nodes = sorted({cpu.numa_node for cpu in logical_cpu_list}) or [0]

                env_key = CpuPlatform.device_control_env_var
                if env_key in os.environ and os.environ[env_key]:
                    visible_nodes = [
                        int(s) for s in os.environ[env_key].split(",") if s.strip()
                    ]
                    filtered = [node for node in allowed_numa_nodes if node in visible_nodes]
                    if filtered:
                        allowed_numa_nodes = filtered

                return allowed_numa_nodes, logical_cpu_list

        CpuPlatform.get_allowed_cpu_core_node_list = classmethod(
            _patched_get_allowed_cpu_core_node_list
        )
        _VLLM_CPU_PATCHED = True

    _ensure_vllm_ipc_support()

    load_start = time.perf_counter()
    llm = LLM(
        model=config.model_id,
        revision=config.revision,
        download_dir=config.download_dir,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=config.tensor_parallel_size,
        enforce_eager=config.enforce_eager,
    )
    load_time = time.perf_counter() - load_start

    if config.do_warmup:
        warmup_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=config.warmup_tokens,
        )
        try:
            llm.generate("Warmup prompt.", warmup_params)
        except Exception:
            pass

    sampling_params = SamplingParams(
        n=1,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        repetition_penalty=config.repetition_penalty,
    )

    with MemoryMonitor() as monitor:
        generate_start = time.perf_counter()
        outputs = llm.generate([config.prompt], sampling_params)
        generate_time = time.perf_counter() - generate_start

    request_output = outputs[0]
    sample_output = request_output.outputs[0]
    completion = sample_output.text.strip()
    prompt_tokens = len(request_output.prompt_token_ids)
    completion_tokens = len(sample_output.token_ids)

    return BenchmarkResult(
        backend="vllm",
        model=config.model_id,
        prompt=config.prompt,
        prompt_tokens=prompt_tokens,
        completion=completion,
        completion_tokens=completion_tokens,
        max_new_tokens=config.max_new_tokens,
        load_time_s=load_time,
        generate_time_s=generate_time,
        peak_memory_bytes=monitor.max_rss_bytes,
        num_threads=config.num_threads,
        parameters={
            "dtype": dtype,
            "revision": config.revision,
            "tensor_parallel_size": config.tensor_parallel_size,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "enforce_eager": config.enforce_eager,
        },
    )


def run_llamacpp_benchmark(config: LlamaCppBenchmarkConfig) -> BenchmarkResult:
    from llama_cpp import Llama

    _ensure_threads_config(config.num_threads)

    load_start = time.perf_counter()
    llm = Llama(
        model_path=config.model_path,
        n_ctx=config.n_ctx,
        n_batch=config.n_batch,
        n_threads=config.num_threads or os.cpu_count() or 1,
        seed=config.seed,
        logits_all=False,
        verbose=False,
    )
    load_time = time.perf_counter() - load_start

    if config.do_warmup:
        try:
            llm(
                "Warmup prompt.",
                max_tokens=config.warmup_tokens,
                temperature=0.0,
            )
        except Exception:
            pass

    with MemoryMonitor() as monitor:
        generate_start = time.perf_counter()
        output = llm(
            config.prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repetition_penalty,
        )
        generate_time = time.perf_counter() - generate_start

    usage = output.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    completion = output["choices"][0]["text"].strip()

    return BenchmarkResult(
        backend="llama.cpp",
        model=config.model_path,
        prompt=config.prompt,
        prompt_tokens=prompt_tokens,
        completion=completion,
        completion_tokens=completion_tokens,
        max_new_tokens=config.max_new_tokens,
        load_time_s=load_time,
        generate_time_s=generate_time,
        peak_memory_bytes=monitor.max_rss_bytes,
        num_threads=config.num_threads or os.cpu_count() or 1,
        parameters={
            "n_ctx": config.n_ctx,
            "n_batch": config.n_batch,
            "seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
    )


def aggregate_results(results: Iterable[BenchmarkResult]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "system": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
        },
        "results": [],
    }
    for result in results:
        payload["results"].append(result.to_dict())
    return payload


def write_results(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def format_results_table(results: Iterable[BenchmarkResult]) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        raise RuntimeError("tabulate package is required to format results.")

    headers = [
        "Backend",
        "Model",
        "Threads",
        "Load (s)",
        "Gen (s)",
        "Tokens",
        "Tok/s",
        "Peak Mem (MiB)",
    ]
    rows: List[List[Any]] = []
    for result in results:
        rows.append(
            [
                result.backend,
                Path(result.model).name if Path(result.model).exists() else result.model,
                result.num_threads or "-",
                f"{result.load_time_s:.2f}",
                f"{result.generate_time_s:.2f}",
                result.completion_tokens,
                f"{result.tokens_per_second:.2f}",
                f"{result.peak_memory_bytes / (1024 ** 2):.1f}",
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="github")
