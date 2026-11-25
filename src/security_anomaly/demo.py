"""Export the small synthetic Zeek example shipped with the package."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .package_resources import resource_ref
from .serialization import OutputWriteError


def _export(args: argparse.Namespace) -> int:
    destination = args.output_dir.expanduser().absolute()
    resources = resource_ref("zeek")
    inputs = {name: resources.joinpath(name).read_bytes() for name in ("fit.jsonl", "heldout.jsonl")}
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
                prepared = Path(staging) / "zeek"
                prepared.mkdir()
                for name, data in inputs.items():
                    (prepared / name).write_bytes(data)
                try:
                    prepared.rename(destination)
                except OSError:
                    if not destination.exists():
                        raise
        # Concurrent and repeated exports succeed only for the same example.
        if not all((destination / name).is_file() and (destination / name).read_bytes() == data
                   for name, data in inputs.items()):
            raise OutputWriteError(f"Choose a new example directory: {destination}")
    except OSError as error:
        raise OutputWriteError(f"Cannot export Zeek example: {error}") from error
    print(json.dumps({"directory": str(destination), "files": list(inputs)}))
    return 0


def register_parser(commands: argparse._SubParsersAction) -> None:
    demo = commands.add_parser("demo", help="export installed synthetic example logs")
    examples = demo.add_subparsers(dest="example", required=True)
    zeek = examples.add_parser("zeek", help="write the Zeek H2 example logs")
    zeek.add_argument("--output-dir", required=True, type=Path, help="new directory for the two logs")
    zeek.set_defaults(handler=_export)
