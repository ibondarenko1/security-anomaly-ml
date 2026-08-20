"""Export deterministic v0.1 contracts from the already-frozen local artifacts.

This command verifies provenance hashes and metadata. It never fits, retrains,
or modifies the model and never reads raw datasets or holdout rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "v2_context_random_forest.joblib"
DEFAULT_FEATURE_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2" / "feature_manifest.json"
)
DEFAULT_ABLATION_METRICS = PROJECT_ROOT / "models" / "v2_ablation_metrics.json"
DEFAULT_FLOW_METRICS = PROJECT_ROOT / "models" / "v2_feb18_flow_metrics.json"
DEFAULT_INCIDENT_METRICS = PROJECT_ROOT / "models" / "v2_feb18_incident_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "contracts"
RESEARCH_BUILDER = PROJECT_ROOT / "src" / "build_temporal_features.py"

FEATURE_CONTRACT_FILENAME = "feature-contract-cicflow-v2-128.json"
MODEL_MANIFEST_FILENAME = "model-manifest-context-rf-v2.json"
EXPECTED_HASHES = {
    "model": "4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72",
    "feature_manifest": "d313c554798945a5ab34a43237dd0e57019c35e550b2ded987bce7039b5778f8",
    "research_builder": "7b12e85df31fab6dcb986ea304ae89390dce395078c7b5e232fa6b391be84c18",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Frozen {label} SHA256 mismatch: expected {expected}, got {actual}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST
    )
    parser.add_argument("--ablation-metrics", type=Path, default=DEFAULT_ABLATION_METRICS)
    parser.add_argument("--flow-metrics", type=Path, default=DEFAULT_FLOW_METRICS)
    parser.add_argument("--incident-metrics", type=Path, default=DEFAULT_INCIDENT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_hash(args.model, EXPECTED_HASHES["model"], "Context model")
    verify_hash(
        args.feature_manifest,
        EXPECTED_HASHES["feature_manifest"],
        "feature manifest",
    )
    verify_hash(
        RESEARCH_BUILDER,
        EXPECTED_HASHES["research_builder"],
        "research feature builder",
    )

    feature_manifest = load_json(args.feature_manifest)
    feature_source = feature_manifest["feature_contract"]
    baseline = feature_source["baseline_v2_features"]
    static = feature_source["context_v2_static_behavioral_features"]
    temporal = feature_source["context_v2_temporal_features"]
    model_features = [*baseline, *static, *temporal]
    if (len(baseline), len(static), len(temporal), len(model_features)) != (
        76,
        9,
        43,
        128,
    ):
        raise RuntimeError("Frozen feature manifest does not contain 76 + 9 + 43 features")

    contract = {
        "schema_version": "feature-contract-v1",
        "feature_contract": "cicflow-v2-128",
        "feature_builder_version": "causal-temporal-v2",
        "model_feature_count": 128,
        "model_features": model_features,
        "feature_groups": {
            "baseline": {
                "count": 76,
                "description": "Numeric CICFlowMeter flow features consumed unchanged.",
                "features": baseline,
            },
            "static_behavioral": {
                "count": 9,
                "description": "Raw numeric ports/protocol plus deterministic port ranges.",
                "features": static,
            },
            "temporal": {
                "count": 43,
                "description": "Causal host/endpoint context from timestamps strictly before t.",
                "features": temporal,
            },
        },
        "input": {
            "format": "CICFlowMeter-compatible rows",
            "required_identity_context_columns": [
                "Src IP",
                "Src Port",
                "Dst IP",
                "Dst Port",
                "Protocol",
                "Timestamp",
            ],
            "optional_preserved_columns": ["Flow ID"],
            "required_model_columns": baseline,
            "labels_required": False,
            "ignored_evaluation_columns": ["Label", "attack_cat", "label"],
        },
        "excluded_from_model": ["Flow ID", "Src IP", "Dst IP", "Timestamp"],
        "derived_definitions": {
            "flow_bytes": feature_source["flow_bytes_definition"],
            "flow_packets": feature_source["flow_packets_definition"],
            "cold_start_seconds_since_value": feature_source[
                "cold_start_seconds_since_value"
            ],
            "port_ranges": feature_source["port_ranges"],
        },
        "temporal_state": {
            "initial_mode": "batch-empty",
            "sort_key": ["Timestamp"],
            "windows_seconds": [10, 60],
            "lower_window_boundary_inclusive": True,
            "history_upper_boundary": "strictly earlier than current timestamp",
            "same_timestamp_policy": (
                "All rows at timestamp t are peers. Every peer is featurized before "
                "the complete peer group updates state."
            ),
            "state_carries_between_batches": False,
        },
        "numeric_policy": {
            "required_input_values_finite": True,
            "model_matrix_dtype": "float32",
            "categorical_encoding": None,
            "scaling": None,
        },
        "provenance": {
            "source_feature_manifest_sha256": EXPECTED_HASHES["feature_manifest"],
            "research_builder_sha256": EXPECTED_HASHES["research_builder"],
            "raw_datasets_included": False,
        },
    }
    contract_path = args.output_dir / FEATURE_CONTRACT_FILENAME
    write_json(contract_path, contract)
    contract_hash = sha256_file(contract_path)

    ablation = load_json(args.ablation_metrics)
    flow = load_json(args.flow_metrics)
    incident = load_json(args.incident_metrics)
    model = joblib.load(args.model)
    if int(model.n_features_in_) != 128 or model.classes_.tolist() != [0, 1]:
        raise RuntimeError("Frozen model interface differs from the release contract")
    frozen_parameter_names = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "max_samples",
        "bootstrap",
        "random_state",
        "class_weight",
        "criterion",
    ]
    model_params = model.get_params(deep=False)
    software = ablation["software"]
    manifest = {
        "schema_version": "model-bundle-manifest-v1",
        "product_version": "0.1.0",
        "model_version": "context-rf-v2",
        "artifact": {
            "filename": "context-rf-v2.joblib",
            "sha256": EXPECTED_HASHES["model"],
            "size_bytes": args.model.stat().st_size,
            "distribution": "GitHub Release asset; intentionally excluded from Git history",
        },
        "serialization": {
            "python_version": "3.13.7",
            "python_major_minor": "3.13",
            "required_package_versions": {
                "scikit-learn": software["scikit_learn"],
                "numpy": software["numpy"],
                "pandas": software["pandas"],
                "duckdb": software["duckdb"],
                "joblib": "1.5.3",
            },
            "runtime_requirements_file": "requirements-runtime.txt",
        },
        "compatibility": {
            "feature_contract": "cicflow-v2-128",
            "feature_contract_sha256": contract_hash,
            "feature_builder_version": "causal-temporal-v2",
            "model_feature_count": 128,
            "incompatible_combinations_must_fail_closed": True,
        },
        "model": {
            "class": "sklearn.ensemble._forest.RandomForestClassifier",
            "classes": [0, 1],
            "score_semantics": (
                "predict_proba[:, 1] attack score; not calibrated real-world probability"
            ),
            "frozen_parameters": {
                name: model_params[name] for name in frozen_parameter_names
            },
        },
        "operational_policy": {
            "flow_alert": {"field": "attack_score", "operator": ">=", "threshold": 0.10},
            "incident_aggregation": {
                "policy": "B",
                "grouping_key": ["src_ip", "dst_ip", "dst_port"],
                "window_seconds": 300,
            },
            "incident_promotion": {
                "field": "max_attack_score",
                "operator": ">=",
                "threshold": 0.25,
            },
        },
        "provenance": {
            "training_capture": "CIC-UNSW-NB15 / 2015-01-22",
            "validation_capture": "CIC-UNSW-NB15 / 2015-02-17",
            "locked_temporal_holdout": "CIC-UNSW-NB15 / 2015-02-18",
            "locked_holdout_never_used_for_selection": True,
            "post_holdout_tuning_performed": False,
            "holdout_summary": {
                "flow_recall": flow["recall"],
                "flow_fpr": flow["fpr"],
                "flow_pr_auc": flow["pr_auc_average_precision"],
                "aggregated_incident_recall": incident["before_promotion"][
                    "incident_recall"
                ],
                "promoted_incident_recall": incident["after_promotion"][
                    "incident_recall"
                ],
                "promoted_incident_precision": incident["after_promotion"][
                    "incident_precision"
                ],
                "verdict": "Acceptable but operationally noisy",
            },
        },
        "redistribution": {
            "repository_code_license": "Apache-2.0",
            "dataset_included": False,
            "dataset_terms": (
                "UNSW-NB15 and CIC-UNSW-NB15 remain subject to their original "
                "publisher licensing and terms; no dataset rows are bundled."
            ),
            "artifact_notice": (
                "The model is provided for research and evaluation without a "
                "production-readiness claim."
            ),
        },
    }
    manifest_path = args.output_dir / MODEL_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    print(f"Feature contract: {contract_path} ({contract_hash})")
    print(f"Model manifest: {manifest_path}")
    print("Frozen model was verified and loaded; it was not modified or retrained.")


if __name__ == "__main__":
    main()
