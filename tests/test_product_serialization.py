from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

from src.security_anomaly.package_resources import (
    FEATURE_CONTRACT_RESOURCE,
    INCIDENT_SCHEMA_RESOURCE,
    MODEL_MANIFEST_RESOURCE,
    resource_bytes,
    resource_json,
)
from src.security_anomaly.aggregation import aggregate_policy_b_alerts
from src.security_anomaly.results import (
    BatchAnalysisResult,
    FlowDetection,
    IncidentDetection,
)
from src.security_anomaly.serialization import (
    IncidentSerializationError,
    OutputWriteError,
    incident_id_v1,
    incident_to_v1,
    jsonl_text,
    promoted_incidents_to_v1,
    write_incidents_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def incident() -> IncidentDetection:
    return IncidentDetection(
        incident_sequence=99,
        first_seen=datetime(2015, 2, 17, 12, 3, 21),
        last_seen=datetime(2015, 2, 17, 12, 4, 12),
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        dst_port=445,
        protocols=(17, 6, 17),
        flow_count=3,
        max_attack_score=0.73,
        mean_attack_score=0.31,
        promoted=True,
        source_row_ids=(7, 4, 9),
    )


def public_value() -> dict[str, object]:
    return incident_to_v1(
        incident(),
        product_version="0.1.0",
        model_version="context-rf-v2",
        feature_contract="cicflow-v2-128",
    )


def test_public_incident_is_valid_against_json_schema() -> None:
    jsonschema.validate(public_value(), resource_json(INCIDENT_SCHEMA_RESOURCE))


def test_schema_rejects_missing_and_unexpected_ground_truth_fields() -> None:
    schema = resource_json(INCIDENT_SCHEMA_RESOURCE)
    missing = public_value()
    missing.pop("incident_id")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, schema)
    for field in ("Label", "label", "attack_cat"):
        unexpected = public_value()
        unexpected[field] = "ground-truth"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(unexpected, schema)


def test_protocols_are_a_sorted_unique_integer_array() -> None:
    value = public_value()
    assert value["protocols"] == [6, 17]
    assert all(type(protocol) is int for protocol in value["protocols"])
    invalid = dict(value)
    invalid["protocols"] = ["6"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, resource_json(INCIDENT_SCHEMA_RESOURCE))


def test_timestamp_does_not_invent_utc() -> None:
    value = public_value()
    assert value["first_seen"] == "2015-02-17T12:03:21"
    assert not str(value["first_seen"]).endswith("Z")


def test_serializer_matches_batch_promoted_incidents() -> None:
    item = incident()
    result = BatchAnalysisResult(
        flows_processed=3,
        flow_alert_count=3,
        incident_count=1,
        promoted_incident_count=1,
        flow_detections=(),
        incidents=(item,),
        promoted_incidents=(item,),
        product_version="0.1.0",
        model_version="context-rf-v2",
        feature_contract="cicflow-v2-128",
        state_mode="batch-empty",
    )
    assert promoted_incidents_to_v1(result) == (public_value(),)


def test_public_scores_are_canonical_across_sub_tolerance_numeric_noise() -> None:
    first = incident_to_v1(
        replace(
            incident(),
            max_attack_score=0.4325909090909091,
            mean_attack_score=0.4183686868686869,
        ),
        product_version="0.1.0",
        model_version="context-rf-v2",
        feature_contract="cicflow-v2-128",
    )
    second = incident_to_v1(
        replace(
            incident(),
            max_attack_score=0.432590909090909,
            mean_attack_score=0.418368686868687,
        ),
        product_version="0.1.0",
        model_version="context-rf-v2",
        feature_contract="cicflow-v2-128",
    )

    assert first["max_attack_score"] == second["max_attack_score"] == 0.432590909091
    assert first["mean_attack_score"] == second["mean_attack_score"] == 0.418368686869


def test_non_promoted_and_non_finite_values_fail_closed() -> None:
    with pytest.raises(IncidentSerializationError, match="promoted-only"):
        incident_to_v1(
            replace(incident(), promoted=False),
            product_version="0.1.0",
            model_version="context-rf-v2",
            feature_contract="cicflow-v2-128",
        )
    with pytest.raises(IncidentSerializationError, match="finite"):
        incident_to_v1(
            replace(incident(), max_attack_score=float("nan")),
            product_version="0.1.0",
            model_version="context-rf-v2",
            feature_contract="cicflow-v2-128",
        )
    with pytest.raises(ValueError):
        jsonl_text([{"score": float("inf")}])


def test_incident_id_is_repeatable_and_independent_of_internal_ordering() -> None:
    item = incident()
    changed_internal = replace(
        item,
        incident_sequence=0,
        source_row_ids=tuple(reversed(item.source_row_ids)),
        protocols=tuple(reversed(item.protocols)),
    )
    assert incident_id_v1(item) == incident_id_v1(item)
    assert incident_id_v1(item) == incident_id_v1(changed_internal)


def test_incident_id_changes_for_policy_key_or_session_time() -> None:
    item = incident()
    assert incident_id_v1(replace(item, dst_port=443)) != incident_id_v1(item)
    assert incident_id_v1(
        replace(item, last_seen=datetime(2015, 2, 17, 12, 4, 13))
    ) != incident_id_v1(item)


def test_equivalent_reordered_flows_produce_same_public_incident_id() -> None:
    flows = (
        FlowDetection(
            source_row_id=8,
            timestamp=datetime(2015, 2, 17, 12, 3, 21),
            flow_id="b",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=50001,
            dst_port=445,
            protocol=17,
            attack_score=0.4,
            is_alert=True,
        ),
        FlowDetection(
            source_row_id=2,
            timestamp=datetime(2015, 2, 17, 12, 4, 12),
            flow_id="a",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=50000,
            dst_port=445,
            protocol=6,
            attack_score=0.5,
            is_alert=True,
        ),
    )
    first = aggregate_policy_b_alerts(
        flows, window_seconds=300, promotion_threshold=0.25
    )[0]
    reordered = aggregate_policy_b_alerts(
        reversed(flows), window_seconds=300, promotion_threshold=0.25
    )[0]
    assert incident_id_v1(first) == incident_id_v1(reordered)


@pytest.mark.parametrize(
    ("repository_name", "resource_name"),
    [
        ("feature-contract-cicflow-v2-128.json", FEATURE_CONTRACT_RESOURCE),
        ("model-manifest-context-rf-v2.json", MODEL_MANIFEST_RESOURCE),
        ("incident-v1.schema.json", INCIDENT_SCHEMA_RESOURCE),
    ],
)
def test_repository_and_package_resources_are_byte_identical(
    repository_name: str, resource_name: str
) -> None:
    assert (ROOT / "contracts" / repository_name).read_bytes() == resource_bytes(
        resource_name
    )


def test_jsonl_is_utf8_strict_and_newline_terminated() -> None:
    output = jsonl_text([public_value()])
    assert output.endswith("\n")
    assert json.loads(output) == public_value()


def test_failed_serialization_does_not_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "incidents.jsonl"
    output.write_text("previous\n", encoding="utf-8")
    with pytest.raises(OutputWriteError):
        write_incidents_jsonl(output, [{"score": float("nan")}])
    assert output.read_text(encoding="utf-8") == "previous\n"
