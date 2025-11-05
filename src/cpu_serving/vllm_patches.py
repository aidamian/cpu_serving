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


def apply_all() -> Tuple[bool, bool]:
    """Apply every available vLLM patch."""

    return patch_cpu_topology(), ensure_vllm_ipc_support()
