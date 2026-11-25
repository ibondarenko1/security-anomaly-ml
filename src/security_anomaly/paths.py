"""Protect local inputs when a command replaces an output file."""

from pathlib import Path

from .serialization import OutputWriteError


def ensure_distinct(destination: Path, *protected: Path) -> None:
    destination = destination.expanduser().resolve()
    for path in protected:
        source = path.expanduser().resolve()
        if destination == source or (
            destination.exists() and source.exists() and destination.samefile(source)
        ):
            raise OutputWriteError(f"Output path must differ from input: {source}")
