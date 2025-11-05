"""Utility helpers for colorized console logging."""

from __future__ import annotations

from typing import Final, Mapping

_COLOR_CODES: Mapping[str, str] = {
    "g": "\033[92m",  # green
    "b": "\033[94m",  # blue
    "y": "\033[93m",  # yellow
    "r": "\033[91m",  # red
    "d": "\033[90m",  # dark grey
}

_RESET: Final[str] = "\033[0m"


def log_color(message: object, color: str = "d") -> None:
    """Pretty-print a message with ANSI colors and flush immediately."""
    text = str(message)
    code = _COLOR_CODES.get(color, "")
    output = f"{code}{text}{_RESET}" if code else text
    print(output, flush=True)

