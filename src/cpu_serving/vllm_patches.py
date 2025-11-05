from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, List, Tuple
from uuid import uuid4

_CPU_TOPOLOGY_PATCHED = False
_IPC_PATCHED = False
_TORCH_THREADS_PATCHED = False
_CPU_PLATFORM_PATCHED = False
_CACHE_ROOT_PATCHED = False


def patch_cpu_topology() -> bool:
    """Ensure vLLM tolerates missing NUMA info from `lscpu -J` output."""

    global _CPU_TOPOLOGY_PATCHED
    if _CPU_TOPOLOGY_PATCHED:
        return False

    try:
        from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo  # type: ignore
    except ImportError:
        return False

    original = getattr(CpuPlatform.get_allowed_cpu_core_node_list, "__func__", None)
    if original is None:
        return False

    def _patched_get_allowed_cpu_core_node_list(cls):  # type: ignore[override]
        try:
            return original(cls)
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
                return original(cls)

            sanitized = re.sub(r'"node"\s*:\s*-', '"node": 0', lscpu_output)
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

            allowed_numa_nodes = sorted(
                {cpu.numa_node for cpu in logical_cpu_list}
            ) or [0]

            env_key = CpuPlatform.device_control_env_var
            if env_key in os.environ and os.environ[env_key]:
                visible_nodes = [
                    int(s) for s in os.environ[env_key].split(",") if s.strip()
                ]
                filtered = [
                    node for node in allowed_numa_nodes if node in visible_nodes
                ]
                if filtered:
                    allowed_numa_nodes = filtered

            return allowed_numa_nodes, logical_cpu_list

    CpuPlatform.get_allowed_cpu_core_node_list = classmethod(  # type: ignore[assignment]
        _patched_get_allowed_cpu_core_node_list
    )
    _CPU_TOPOLOGY_PATCHED = True
    return True


def ensure_vllm_ipc_support() -> bool:
    """Force vLLM ZeroMQ helpers to fall back to TCP when IPC bind is blocked."""

    global _IPC_PATCHED
    if _IPC_PATCHED:
        return False

    try:
        import zmq  # type: ignore
        from vllm import utils as vllm_utils  # type: ignore
    except ImportError:
        return False

    ctx = zmq.Context()  # type: ignore[attr-defined]
    socket_path = Path(tempfile.gettempdir()) / f"cpu-serving-zmq-{uuid4().hex}"
    ipc_uri = f"ipc://{socket_path}"
    original_get_ipc = getattr(vllm_utils, "get_open_zmq_ipc_path", None)
    sock = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
    patched = False

    try:
        sock.bind(ipc_uri)
    except zmq.ZMQError as exc:  # type: ignore[attr-defined]
        if exc.errno not in {getattr(zmq, "EPERM", errno.EPERM), errno.EPERM}:
            return False

        def _tcp_uri() -> str:
            try:
                port = vllm_utils.get_open_port()
                return f"tcp://127.0.0.1:{port}"
            except OSError:
                if callable(original_get_ipc):
                    return original_get_ipc()
                return ipc_uri

        modules_to_patch: List[Any] = [vllm_utils]
        for import_path in (
            "vllm.v1.utils",
            "vllm.v1.engine.utils",
            "vllm.distributed.device_communicators.shm_broadcast",
        ):
            try:
                module = import_module(import_path)
            except ImportError:
                continue
            modules_to_patch.append(module)

        for module in modules_to_patch:
            if hasattr(module, "get_open_zmq_ipc_path"):
                module.get_open_zmq_ipc_path = _tcp_uri  # type: ignore[assignment]

        patched = True
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

    if patched:
        _IPC_PATCHED = True
    return patched


def ensure_torch_thread_binding_stub() -> bool:
    """Provide a no-op torch operator when custom CPU thread binding is absent."""

    global _TORCH_THREADS_PATCHED
    if _TORCH_THREADS_PATCHED:
        return False

    try:
        import torch
    except ImportError:
        return False

    namespace = getattr(torch.ops, "_C_utils", None)
    if namespace is None or hasattr(namespace, "init_cpu_threads_env"):
        return False

    def _stub(cpu_spec: str) -> None:
        return None

    setattr(namespace, "init_cpu_threads_env", _stub)
    _TORCH_THREADS_PATCHED = True
    return True


def ensure_cpu_platform() -> bool:
    """Ensure the global platform object behaves like the CPU backend."""

    global _CPU_PLATFORM_PATCHED
    if _CPU_PLATFORM_PATCHED:
        return False

    try:
        from vllm import platforms as vllm_platforms  # type: ignore
        from vllm.platforms.cpu import CpuPlatform  # type: ignore
    except ImportError:
        return False

    current = vllm_platforms.current_platform
    if current.is_cpu():
        _CPU_PLATFORM_PATCHED = True
        return True

    current.__class__ = CpuPlatform  # type: ignore[misc]
    current.device_type = "cpu"
    current.device_name = "cpu"
    current.dispatch_key = "CPU"
    current.dist_backend = "gloo"
    current._enum = CpuPlatform._enum  # type: ignore[attr-defined]
    _CPU_PLATFORM_PATCHED = True
    return True


def ensure_vllm_cache_root() -> bool:
    """Point vLLM cache lookups at a writable workspace directory."""

    global _CACHE_ROOT_PATCHED
    if _CACHE_ROOT_PATCHED:
        return False

    if os.environ.get("VLLM_CACHE_ROOT"):
        _CACHE_ROOT_PATCHED = True
        return False

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except Exception:
        repo_root = Path.cwd()

    cache_root = repo_root / ".cache" / "vllm"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    os.environ["VLLM_CACHE_ROOT"] = str(cache_root)
    _CACHE_ROOT_PATCHED = True
    return True


def apply_all() -> Tuple[bool, bool, bool, bool, bool]:
    """Apply every available vLLM patch."""

    return (
        ensure_vllm_cache_root(),
        ensure_cpu_platform(),
        patch_cpu_topology(),
        ensure_vllm_ipc_support(),
        ensure_torch_thread_binding_stub(),
    )
