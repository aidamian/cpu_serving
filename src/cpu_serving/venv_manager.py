from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


class VirtualEnvError(RuntimeError):
    """Raised when a virtual environment cannot be prepared."""


@dataclass(frozen=True)
class EnvSpec:
    key: str
    name: str
    requirements: Path
    python_candidates: Tuple[str, ...]


@dataclass(frozen=True)
class VirtualEnvHandle:
    spec: EnvSpec
    path: Path
    python: Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_DIR = _REPO_ROOT / "envs"

_BACKEND_ALIASES: Dict[str, str] = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "vllm": "vllm",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "llama": "llamacpp",
}

_ENV_SPECS: Dict[str, EnvSpec] = {
    "huggingface": EnvSpec(
        key="huggingface",
        name="venv-hf",
        requirements=_ENV_DIR / "requirements-hf.txt",
        python_candidates=("python3.12", "python3"),
    ),
    "vllm": EnvSpec(
        key="vllm",
        name="venv-vllm",
        requirements=_ENV_DIR / "requirements-vllm.txt",
        python_candidates=("python3.13", "python3.12", "python3"),
    ),
    "llamacpp": EnvSpec(
        key="llamacpp",
        name="venv-llamacpp",
        requirements=_ENV_DIR / "requirements-llamacpp.txt",
        python_candidates=("python3.12", "python3"),
    ),
}


def available_backends() -> Iterable[str]:
    return tuple(_ENV_SPECS.keys())


def resolve_backend(name: str) -> str:
    try:
        return _BACKEND_ALIASES[name.lower()]
    except KeyError as exc:
        raise VirtualEnvError(f"Unknown backend '{name}'.") from exc


def get_spec(backend: str) -> EnvSpec:
    canonical = resolve_backend(backend)
    return _ENV_SPECS[canonical]


def virtualenv_home() -> Path:
    base = os.environ.get("VIRTUALENV_HOME")
    if base:
        return Path(base).expanduser().resolve()
    return (_REPO_ROOT / ".venvs").resolve()


def _find_python(candidates: Iterable[str]) -> Path:
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return Path(path).resolve()
    raise VirtualEnvError(
        f"Unable to locate a Python interpreter from candidates: {', '.join(candidates)}"
    )


def _venv_python_path(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def ensure_virtualenv(
    backend: str,
    *,
    sync_dependencies: bool = True,
    reinstall: bool = False,
    upgrade: bool = False,
) -> VirtualEnvHandle:
    spec = get_spec(backend)
    home = virtualenv_home()
    home.mkdir(parents=True, exist_ok=True)

    python_exe = _find_python(spec.python_candidates)
    venv_path = (home / spec.name).resolve()

    if reinstall and venv_path.exists():
        shutil.rmtree(venv_path)

    python_bin = _venv_python_path(venv_path)
    if not python_bin.exists():
        subprocess.run([str(python_exe), "-m", "venv", str(venv_path)], check=True)

    if sync_dependencies:
        _install_requirements(python_bin, spec.requirements, upgrade=upgrade)
        _install_project(python_bin)

    return VirtualEnvHandle(spec=spec, path=venv_path, python=python_bin)


def _install_requirements(python_bin: Path, requirements: Path, *, upgrade: bool) -> None:
    if not requirements.exists():
        raise VirtualEnvError(f"Requirements file '{requirements}' not found.")
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_bin),
        "--requirement",
        str(requirements),
        "--index-strategy",
        "unsafe-best-match",
    ]
    if upgrade:
        cmd.append("--upgrade")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def _install_project(python_bin: Path) -> None:
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_bin),
        "--editable",
        str(_REPO_ROOT),
    ]
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
