"""Build a local GitHub Release bundle for the frozen Context model.

The binary remains outside Git history. The command verifies the trusted model
hash and contract compatibility before copying any release content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security_anomaly.model_bundle import FrozenModelBundle  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models" / "v2_context_random_forest.joblib"
DEFAULT_MANIFEST = PROJECT_ROOT / "contracts" / "model-manifest-context-rf-v2.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Generated release directory; use a path excluded from Git history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing release directory: {args.output_dir}"
        )
    bundle = FrozenModelBundle.load(args.model, args.manifest, args.contract)
    args.output_dir.mkdir(parents=True)
    output_model = args.output_dir / bundle.manifest["artifact"]["filename"]
    output_manifest = args.output_dir / args.manifest.name
    output_contract = args.output_dir / args.contract.name
    shutil.copy2(args.model, output_model)
    shutil.copy2(args.manifest, output_manifest)
    shutil.copy2(args.contract, output_contract)
    sums = {
        path.name: sha256_file(path)
        for path in (output_model, output_manifest, output_contract)
    }
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(sums, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(args.output_dir), "sha256": sums}, indent=2))


if __name__ == "__main__":
    main()
