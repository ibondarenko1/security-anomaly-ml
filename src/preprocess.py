"""Prepare UNSW-NB15 feature matrices without training a model.

The official training set is split into development train/validation subsets.
The official testing set remains a separate final holdout. The preprocessor is
fit only on the development training subset, and no transformed data is saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OFFICIAL_TRAINING_FILE = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
OFFICIAL_TEST_FILE = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET_COLUMN = "label"
EXCLUDED_COLUMNS = ("id", "attack_cat")
CATEGORICAL_COLUMNS = ("proto", "service", "state")
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


@dataclass
class DevelopmentRawSplit:
    """Raw train/validation split created only from official training data."""

    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series


@dataclass
class DevelopmentPreprocessingResult:
    """In-memory preprocessing outputs that never access the official test."""

    preprocessor: ColumnTransformer
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    X_train: Any
    X_validation: Any
    numeric_features: list[str]
    categorical_features: list[str]
    learned_categories: dict[str, list[str]]
    validation_unknown_categories: dict[str, list[str]]


@dataclass
class PreprocessingResult:
    """All in-memory outputs from the preprocessing stage."""

    preprocessor: ColumnTransformer
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    X_train: Any
    X_validation: Any
    X_test: Any
    numeric_features: list[str]
    categorical_features: list[str]
    learned_categories: dict[str, list[str]]
    validation_unknown_categories: dict[str, list[str]]
    test_unknown_categories: dict[str, list[str]]


def load_official_training() -> pd.DataFrame:
    """Load only the official training CSV and attach in-memory provenance."""
    if not OFFICIAL_TRAINING_FILE.is_file():
        raise FileNotFoundError(
            f"Required official training file not found: {OFFICIAL_TRAINING_FILE}"
        )
    official_training = pd.read_csv(OFFICIAL_TRAINING_FILE)
    official_training.index = pd.RangeIndex(
        len(official_training), name="official_training_row_index"
    )
    return official_training


def load_official_test() -> pd.DataFrame:
    """Load only the official final-test CSV and attach provenance."""
    if not OFFICIAL_TEST_FILE.is_file():
        raise FileNotFoundError(
            f"Required official test file not found: {OFFICIAL_TEST_FILE}"
        )
    official_test = pd.read_csv(OFFICIAL_TEST_FILE)
    official_test.index = pd.RangeIndex(
        len(official_test), name="official_test_row_index"
    )
    return official_test


def load_official_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both original CSV files for the explicit inspection stage."""
    official_training = load_official_training()
    official_test = load_official_test()
    return official_training, official_test


def assert_binary_target(target: pd.Series, name: str) -> None:
    """Require a complete binary target containing only zero and one."""
    values = set(target.dropna().unique().tolist())
    assert values == {0, 1}, f"{name} target values are {values}, expected {{0, 1}}"
    assert not target.isna().any(), f"{name} target contains missing values"


def separate_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove targets and the technical identifier from model input."""
    required_columns = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

    target = data[TARGET_COLUMN].copy()
    features = data.drop(columns=[TARGET_COLUMN, *EXCLUDED_COLUMNS]).copy()
    assert not required_columns.intersection(features.columns), (
        "An excluded or target column remains in X"
    )
    return features, target


def identify_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names in source-column order."""
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    ungrouped_features = [
        column
        for column in features.columns
        if column not in numeric_features and column not in categorical_features
    ]

    assert not ungrouped_features, f"Unsupported feature dtypes: {ungrouped_features}"
    assert categorical_features == list(CATEGORICAL_COLUMNS), (
        f"Unexpected categorical features: {categorical_features}"
    )
    assert len(numeric_features) + len(categorical_features) == features.shape[1]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build numeric scaling and unknown-safe categorical encoding."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def assert_finite_matrix(matrix: Any, name: str) -> None:
    """Check dense values or the stored values of a sparse matrix."""
    values = matrix.data if hasattr(matrix, "tocsr") else np.asarray(matrix)
    assert not np.isnan(values).any(), f"{name} contains NaN values"
    assert np.isfinite(values).all(), f"{name} contains infinite values"


def transform_network_flows(
    preprocessor: ColumnTransformer, flows: pd.DataFrame
) -> Any:
    """Transform raw flow rows using an already-fitted preprocessor."""
    required_features = preprocessor.feature_names_in_.tolist()
    missing_features = set(required_features) - set(flows.columns)
    if missing_features:
        raise ValueError(
            f"Flow data is missing required features: {sorted(missing_features)}"
        )

    transformed = preprocessor.transform(flows.loc[:, required_features])
    assert_finite_matrix(transformed, "transformed flow matrix")
    return transformed


def get_learned_categories(
    preprocessor: ColumnTransformer, categorical_features: list[str]
) -> dict[str, list[str]]:
    """Map every categorical feature to its fitted OneHotEncoder values."""
    encoder = preprocessor.named_transformers_["categorical"]
    return {
        column: categories.astype(str).tolist()
        for column, categories in zip(
            categorical_features, encoder.categories_, strict=True
        )
    }


def find_unknown_categories(
    features: pd.DataFrame, learned_categories: dict[str, list[str]]
) -> dict[str, list[str]]:
    """List values that OneHotEncoder will safely encode as all-zero blocks."""
    unknown: dict[str, list[str]] = {}
    for column, learned in learned_categories.items():
        observed = set(features[column].dropna().astype(str).unique().tolist())
        unknown[column] = sorted(observed - set(learned))
    return unknown


def prepare_development_split() -> DevelopmentRawSplit:
    """Create the deterministic raw split without fitting preprocessing."""
    official_training = load_official_training()
    training_features, training_target = separate_features_and_target(
        official_training
    )
    assert_binary_target(training_target, "official training")

    (
        X_train_raw,
        X_validation_raw,
        y_train,
        y_validation,
    ) = train_test_split(
        training_features,
        training_target,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=training_target,
    )

    assert_binary_target(y_train, "development training")
    assert_binary_target(y_validation, "validation")

    excluded = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    for name, features in (
        ("train", X_train_raw),
        ("validation", X_validation_raw),
    ):
        assert not excluded.intersection(features.columns), (
            f"Excluded columns remain in {name} X"
        )

    assert X_train_raw.index.intersection(X_validation_raw.index).empty, (
        "Train and validation row indexes overlap"
    )
    assert len(X_train_raw) + len(X_validation_raw) == len(official_training)
    assert set(X_train_raw.index).union(X_validation_raw.index) == set(
        official_training.index
    )
    assert X_train_raw.index.name == "official_training_row_index"
    assert X_validation_raw.index.name == "official_training_row_index"

    return DevelopmentRawSplit(
        X_train_raw=X_train_raw,
        X_validation_raw=X_validation_raw,
        y_train=y_train,
        y_validation=y_validation,
    )


def prepare_development_data() -> DevelopmentPreprocessingResult:
    """Fit and transform train/validation without reading the official test."""
    split = prepare_development_split()
    X_train_raw = split.X_train_raw
    X_validation_raw = split.X_validation_raw
    y_train = split.y_train
    y_validation = split.y_validation

    numeric_features, categorical_features = identify_feature_groups(X_train_raw)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # This is the only fit call. Validation and official test data are never
    # passed to fit or fit_transform.
    preprocessor.fit(X_train_raw)

    scaler = preprocessor.named_transformers_["numeric"]
    assert int(scaler.n_samples_seen_) == len(X_train_raw), (
        "Numeric preprocessor was not fit on exactly the training subset"
    )

    learned_categories = get_learned_categories(
        preprocessor, categorical_features
    )
    assert "-" in learned_categories["service"], (
        'service="-" was not retained as a learned category'
    )

    validation_unknown_categories = find_unknown_categories(
        X_validation_raw, learned_categories
    )

    X_train = transform_network_flows(preprocessor, X_train_raw)
    X_validation = transform_network_flows(preprocessor, X_validation_raw)

    output_feature_count = len(preprocessor.get_feature_names_out())
    assert X_train.shape == (len(X_train_raw), output_feature_count)
    assert X_validation.shape == (len(X_validation_raw), output_feature_count)
    assert_finite_matrix(X_train, "X_train")
    assert_finite_matrix(X_validation, "X_validation")

    return DevelopmentPreprocessingResult(
        preprocessor=preprocessor,
        X_train_raw=X_train_raw,
        X_validation_raw=X_validation_raw,
        y_train=y_train,
        y_validation=y_validation,
        X_train=X_train,
        X_validation=X_validation,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        learned_categories=learned_categories,
        validation_unknown_categories=validation_unknown_categories,
    )


def prepare_datasets() -> PreprocessingResult:
    """Also transform the official test for the explicit preprocessing report."""
    development = prepare_development_data()
    official_test = load_official_test()
    X_test_raw, y_test = separate_features_and_target(official_test)

    assert_binary_target(y_test, "final test")
    assert development.X_train_raw.columns.tolist() == X_test_raw.columns.tolist()
    assert X_test_raw.index.name == "official_test_row_index"
    assert len(X_test_raw) == len(official_test)

    excluded = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    assert not excluded.intersection(X_test_raw.columns), (
        "Excluded columns remain in official test X"
    )

    test_unknown_categories = find_unknown_categories(
        X_test_raw, development.learned_categories
    )
    X_test = transform_network_flows(development.preprocessor, X_test_raw)
    output_feature_count = len(development.preprocessor.get_feature_names_out())
    assert X_test.shape == (len(X_test_raw), output_feature_count)
    assert_finite_matrix(X_test, "X_test")

    return PreprocessingResult(
        preprocessor=development.preprocessor,
        X_train_raw=development.X_train_raw,
        X_validation_raw=development.X_validation_raw,
        X_test_raw=X_test_raw,
        y_train=development.y_train,
        y_validation=development.y_validation,
        y_test=y_test,
        X_train=development.X_train,
        X_validation=development.X_validation,
        X_test=X_test,
        numeric_features=development.numeric_features,
        categorical_features=development.categorical_features,
        learned_categories=development.learned_categories,
        validation_unknown_categories=development.validation_unknown_categories,
        test_unknown_categories=test_unknown_categories,
    )


def print_label_distribution(target: pd.Series) -> None:
    """Print binary target counts and percentages."""
    counts = target.value_counts().sort_index()
    report = pd.DataFrame(
        {
            "count": counts,
            "percentage": (counts / len(target) * 100).round(4),
        }
    )
    print(report.to_string())


def print_report(result: PreprocessingResult) -> None:
    """Print the required preprocessing-stage report."""
    print("UNSW-NB15 PREPROCESSING REPORT")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Validation fraction: {VALIDATION_SIZE:.0%}")
    print(f"Raw training subset size: {len(result.X_train_raw):,}")
    print(f"Validation subset size: {len(result.X_validation_raw):,}")
    print(f"Official final test size: {len(result.X_test_raw):,}")

    for name, target in (
        ("train", result.y_train),
        ("validation", result.y_validation),
        ("final test", result.y_test),
    ):
        print(f"\nLabel distribution - {name}:")
        print_label_distribution(target)

    print(f"\nRaw input feature count: {result.X_train_raw.shape[1]}")
    print(f"Numeric feature count: {len(result.numeric_features)}")
    print(f"Categorical feature count: {len(result.categorical_features)}")
    print(
        "Feature count after one-hot encoding: "
        f"{len(result.preprocessor.get_feature_names_out())}"
    )

    print("\nLearned categories:")
    for column in result.categorical_features:
        print(f"{column} ({len(result.learned_categories[column])}):")
        print(result.learned_categories[column])

    print("\nUnknown categories handled with handle_unknown='ignore':")
    print(f"validation: {result.validation_unknown_categories}")
    print(f"official test: {result.test_unknown_categories}")

    print("\nTransformed matrix shapes:")
    print(f"X_train: {result.X_train.shape}")
    print(f"X_validation: {result.X_validation.shape}")
    print(f"X_test: {result.X_test.shape}")

    print("\nExplicit confirmations:")
    print("- Preprocessor fit data: development training subset only")
    print("- Validation fit usage: none")
    print("- Official test feature fit usage: none")
    print("- Official test label fit usage: none")
    print("- Model training: none")
    print("- Transformed CSV files written: none")


def main() -> None:
    """Run preprocessing checks and print the in-memory result summary."""
    result = prepare_datasets()
    print_report(result)


if __name__ == "__main__":
    main()
