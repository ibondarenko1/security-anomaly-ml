from __future__ import annotations

import pandas as pd
import pytest

from src.security_anomaly.contracts import FeatureContract


@pytest.fixture
def product_contract() -> FeatureContract:
    return FeatureContract.load()


@pytest.fixture
def valid_flow_frame(product_contract: FeatureContract) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        name: [0.0, 1.0] for name in product_contract.baseline_features
    }
    data.update(
        {
            "Flow ID": ["fixture-0", "fixture-1"],
            "Src IP": ["10.0.0.1", "10.0.0.1"],
            "Src Port": [50000, 50001],
            "Dst IP": ["10.0.0.2", "10.0.0.2"],
            "Dst Port": [443, 443],
            "Protocol": [6, 17],
            "Timestamp": [
                "17/02/2015 08:00:00 PM",
                "17/02/2015 08:00:01 PM",
            ],
        }
    )
    data["Total Fwd Packet"] = [1, 2]
    data["Total Bwd packets"] = [1, 2]
    data["Total Length of Fwd Packet"] = [100, 200]
    data["Total Length of Bwd Packet"] = [50, 75]
    return pd.DataFrame(data)
