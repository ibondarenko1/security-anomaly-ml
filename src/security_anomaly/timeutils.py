"""Interpret wall-clock capture times in an explicitly selected source zone."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .validation import InputContractError


def source_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise InputContractError(f"Unknown source timezone: {name}") from error


def source_utc(naive: datetime, zone: ZoneInfo) -> datetime:
    if naive.tzinfo is not None:
        raise InputContractError("Expected a timezone-naive source timestamp")
    candidates = set()
    for fold in (0, 1):
        utc = naive.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if utc.astimezone(zone).replace(tzinfo=None) == naive:
            candidates.add(utc)
    if len(candidates) != 1:
        raise InputContractError(
            f"Ambiguous or nonexistent local time in {zone}: {naive}"
        )
    return candidates.pop()
