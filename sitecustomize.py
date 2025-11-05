"""Project-wide startup hooks for python interpreters.

Applying our vLLM patches here ensures that subprocesses spawned via the
`spawn` start method pick them up automatically.
"""

try:
    from cpu_serving.vllm_patches import apply_all
except Exception:
    apply_all = None  # type: ignore[assignment]

if apply_all is not None:  # pragma: no cover - import-time side effect
    try:
        apply_all()
    except Exception:
        # Never block interpreter startup; the runtime patches are best effort.
        pass
