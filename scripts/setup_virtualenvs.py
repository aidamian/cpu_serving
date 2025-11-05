#!/usr/bin/env python
"""Create or update per-backend virtual environments."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from cpu_serving.console import log_color
from cpu_serving.venv_manager import (
    VirtualEnvError,
    available_backends,
    ensure_virtualenv,
    resolve_backend,
)


def _iter_backends(names: Iterable[str]) -> Iterable[str]:
    if not names:
        yield from available_backends()
        return
    for name in names:
        yield resolve_backend(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare isolated virtual environments for each backend."
    )
    parser.add_argument(
        "backends",
        nargs="*",
        help="Optional list of backends to prepare (hf, vllm, llamacpp). Defaults to all.",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Remove existing virtual environments before recreating them.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Create the venv if missing but skip dependency installation.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade already-installed dependencies to the latest allowed versions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = list(dict.fromkeys(_iter_backends(args.backends)))
    if not backends:
        print("No backends selected.", file=sys.stderr)
        return 1

    for backend in backends:
        log_color(f"Preparing backend '{backend}'...", "b")
        try:
            handle = ensure_virtualenv(
                backend,
                sync_dependencies=not args.skip_sync,
                reinstall=args.reinstall,
                upgrade=args.upgrade,
            )
        except VirtualEnvError as exc:
            print(f"Failed to prepare backend '{backend}': {exc}", file=sys.stderr)
            return 2

        log_color(f"  venv path : {handle.path}", "d")
        log_color(f"  interpreter : {handle.python}", "d")
        if args.skip_sync:
            log_color("  dependencies were not updated (--skip-sync).", "y")

    return 0


if __name__ == "__main__":
    sys.exit(main())
