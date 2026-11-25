from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.security_anomaly import zeek


FIXTURES = Path(__file__).parent / "fixtures" / "zeek"


def _profile_and_record() -> tuple[zeek.ZeekProfile, zeek.ZeekRecord]:
    profile = zeek.fit_profile(
        zeek.read_zeek_jsonl(FIXTURES / "fit.jsonl"),
        sensor="explain-sensor",
    )
    record = zeek.read_zeek_jsonl(FIXTURES / "heldout.jsonl").complete_records[0]
    return profile, record


def test_explanation_reports_zero_mad_floor_and_human_reason() -> None:
    profile, record = _profile_and_record()
    altered = replace(
        record,
        features={**record.features, "resp_pkts": 6.0},
    )

    explanation = zeek.explain_score(altered, profile)

    assert [item["feature"] for item in explanation] == ["resp_pkts"]
    item = explanation[0]
    assert item["observed"] == 6.0
    assert item["reference_median"] == 5.0
    assert item["absolute_delta"] == 1.0
    assert item["reference_mad"] == 0.0
    assert item["mad_status"] == "zero"
    assert item["scale_floor_applied"] is True
    assert item["scale_before_floor"] == 0.0
    assert "reference MAD 0 (zero)" in item["reason"]
    assert "floor 1e-09 applied" in item["reason"]


def test_explanation_distinguishes_tiny_nonzero_mad_from_zero() -> None:
    profile, record = _profile_and_record()
    assert profile.mad is not None
    mad = {**profile.mad, "resp_pkts": 1e-12}
    scale = {**profile.scale, "resp_pkts": zeek.MAD_SCALE_FLOOR}
    tiny_profile = replace(profile, mad=mad, scale=scale)
    altered = replace(record, features={**record.features, "resp_pkts": 6.0})

    item = zeek.explain_score(altered, tiny_profile)[0]

    assert item["feature"] == "resp_pkts"
    assert item["reference_mad"] == pytest.approx(1e-12)
    assert item["mad_status"] == "tiny_nonzero"
    assert item["scale_floor_applied"] is True
    assert item["scale_before_floor"] == pytest.approx(1.4826e-12)


def test_explanation_keeps_all_tied_strongest_features_in_profile_order() -> None:
    profile, record = _profile_and_record()
    altered = replace(
        record,
        features={**record.features, "orig_pkts": 5.0, "resp_pkts": 6.0},
    )

    explanation = zeek.explain_score(altered, profile)

    assert [item["feature"] for item in explanation] == ["orig_pkts", "resp_pkts"]
    assert explanation[0]["deviation"] == explanation[1]["deviation"]


def test_legacy_floor_profile_is_explicitly_unknown() -> None:
    profile, record = _profile_and_record()
    legacy_profile = replace(profile, mad=None)
    altered = replace(record, features={**record.features, "resp_pkts": 6.0})

    item = zeek.explain_score(altered, legacy_profile)[0]

    assert item["reference_mad"] is None
    assert item["mad_status"] == "unknown_legacy_floor"
    assert "unknown (legacy profile at floor)" in item["reason"]


@pytest.mark.parametrize("substitution", [999.0, 1e-12])
def test_modified_raw_mad_is_detected_even_with_unchanged_scoring_scale(substitution):
    profile, _ = _profile_and_record()
    payload = profile.to_dict()
    assert zeek.ZeekProfile.from_dict(payload) == profile
    payload["mad"]["resp_pkts"] = substitution
    with pytest.raises(zeek.ZeekProfileError, match="MAD"):
        zeek.ZeekProfile.from_dict(payload)
    del payload["mad"], payload["mad_sha256"]
    legacy = zeek.ZeekProfile.from_dict(payload)
    assert legacy.fingerprint == profile.fingerprint and legacy.mad is None
