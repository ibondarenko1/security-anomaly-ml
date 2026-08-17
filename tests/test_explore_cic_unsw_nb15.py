from __future__ import annotations

import pandas as pd

from src.explore_cic_unsw_nb15 import (
    TIMESTAMP_FORMAT,
    datetime_to_epoch_seconds,
)


def test_datetime_to_epoch_seconds_preserves_2015_dates() -> None:
    timestamps = pd.to_datetime(
        pd.Series(
            [
                "22/01/2015 07:49:41 AM",
                "18/02/2015 08:29:24 AM",
            ]
        ),
        format=TIMESTAMP_FORMAT,
    )

    seconds = datetime_to_epoch_seconds(timestamps)
    round_trip = pd.to_datetime(seconds, unit="s")

    pd.testing.assert_series_equal(
        round_trip.reset_index(drop=True),
        timestamps.reset_index(drop=True),
        check_dtype=False,
    )
