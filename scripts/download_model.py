#!/usr/bin/env python
"""Download and cache Hugging Face model artifacts locally."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if env_token:
        return env_token
    raise SystemExit(
        "A Hugging Face token is required. "
        "Set HUGGINGFACE_HUB_TOKEN / HF_TOKEN in the environment or pass --token."
    )


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "--")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model weights into the local cache."
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-1B",
        help="Repository ID on Hugging Face (default: meta-llama/Llama-3.2-1B).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/commit for the repository.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory (defaults to HF_HOME or .hf_cache).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional output directory to materialize the snapshot.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Restrict downloads to files matching the pattern (can be repeated).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Skip files matching the pattern (can be repeated).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token (falls back to env vars if omitted).",
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        help="Use symlinks in the cache instead of hard copies.",
    )
    return parser.parse_args(argv)


def download_model(args: argparse.Namespace) -> Path:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # pragma: no cover - older hub versions
        try:
            from huggingface_hub.utils import HfHubHTTPError  # type: ignore
        except ImportError:
            HfHubHTTPError = Exception  # type: ignore

    token = _resolve_token(args.token)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else Path(os.environ.get("HF_HOME", Path.cwd() / ".hf_cache")).resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir: Path | None
    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser().resolve()
    else:
        local_dir = (Path.cwd() / "models" / _slugify_model_id(args.model_id)).resolve()

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=args.use_symlinks,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
            token=token,
            resume_download=True,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(f"Failed to download '{args.model_id}': {exc}") from exc
    except Exception as exc:  # pragma: no cover - generic download errors
        raise SystemExit(f"Failed to download '{args.model_id}': {exc}") from exc

    return Path(snapshot_path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot_path = download_model(args)
    print(f"Snapshot ready at: {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
