"""Inspect the original UNSW-NB15 predefined train and test CSV files.

This module is intentionally read-only: it does not clean, transform, encode,
scale, split, or save the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_FILES = (
    RAW_DATA_DIR / "UNSW_NB15_training-set.csv",
    RAW_DATA_DIR / "UNSW_NB15_testing-set.csv",
)
TARGET_COLUMNS = ("label", "attack_cat")


def print_section(title: str) -> None:
    """Print a clear report section heading."""
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def find_identifier_like_columns(data: pd.DataFrame) -> list[str]:
    """Identify obvious row identifiers without removing them."""
    identifier_names = {"id", "index", "row_id", "record_id", "uuid"}
    candidates: list[str] = []

    for column in data.columns:
        normalized_name = column.strip().lower()
        is_identifier_name = (
            normalized_name in identifier_names
            or normalized_name.endswith("_id")
        )
        if is_identifier_name:
            candidates.append(column)

    return candidates


def print_value_distribution(data: pd.DataFrame, column: str) -> None:
    """Print counts and percentages for all values in a column."""
    counts = data[column].value_counts(dropna=False)
    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": (counts / len(data) * 100).round(4),
        }
    )
    print(distribution.to_string())


def inspect_dataset(path: Path) -> pd.DataFrame:
    """Load one original CSV and print the requested inspection report."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required official dataset file was not found: {path}"
        )

    data = pd.read_csv(path)
    missing_targets = [column for column in TARGET_COLUMNS if column not in data]
    if missing_targets:
        raise ValueError(f"Missing expected target columns: {missing_targets}")

    identifier_columns = find_identifier_like_columns(data)
    predictor_columns = [
        column
        for column in data.columns
        if column not in TARGET_COLUMNS and column not in identifier_columns
    ]
    numeric_features = data[predictor_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = data[predictor_columns].select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    other_features = [
        column
        for column in predictor_columns
        if column not in numeric_features and column not in categorical_features
    ]

    print_section(f"FILE: {path.name}")
    print(f"Path: {path}")
    print(f"Rows: {data.shape[0]:,}")
    print(f"Columns: {data.shape[1]:,}")
    print(f"File size: {path.stat().st_size:,} bytes")

    print("\nComplete column list:")
    for position, column in enumerate(data.columns, start=1):
        print(f"{position:>2}. {column}")

    print("\nData types:")
    print(data.dtypes.astype(str).to_string())

    print("\nFirst 3 real rows (unmodified):")
    print(data.head(3).to_string(index=False))

    missing_counts = data.isna().sum()
    missing_report = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percentage": (missing_counts / len(data) * 100).round(4),
        }
    )
    print("\nMissing values by column:")
    print(missing_report.to_string())
    print(f"Total missing cells: {int(missing_counts.sum()):,}")

    duplicate_rows = int(data.duplicated().sum())
    columns_without_identifiers = [
        column for column in data.columns if column not in identifier_columns
    ]
    duplicates_without_identifiers = int(
        data.duplicated(subset=columns_without_identifiers).sum()
    )
    print("\nDuplicate rows:")
    print(f"Exact duplicates across all columns: {duplicate_rows:,}")
    print(
        "Duplicates when identifier-like columns are ignored: "
        f"{duplicates_without_identifiers:,}"
    )

    print("\nTarget column values:")
    print(f"label: {sorted(data['label'].dropna().unique().tolist())}")
    print(
        "attack_cat: "
        f"{sorted(data['attack_cat'].dropna().astype(str).unique().tolist())}"
    )

    print("\nBinary label distribution (0 = normal, 1 = attack/anomalous):")
    print_value_distribution(data, "label")

    print("\nAttack category distribution:")
    print_value_distribution(data, "attack_cat")

    label_category_crosstab = pd.crosstab(
        data["attack_cat"], data["label"], dropna=False
    )
    inconsistent_targets = int(
        (
            ((data["label"] == 0) & (data["attack_cat"] != "Normal"))
            | ((data["label"] == 1) & (data["attack_cat"] == "Normal"))
        ).sum()
    )
    print("\nConsistency of attack_cat and label:")
    print(label_category_crosstab.to_string())
    print(f"Inconsistent target pairs: {inconsistent_targets:,}")

    print("\nFeature groups (targets and obvious identifiers are listed separately):")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(
        f"Categorical features ({len(categorical_features)}): "
        f"{categorical_features}"
    )
    print(f"Other feature types ({len(other_features)}): {other_features}")
    print(f"Target columns ({len(TARGET_COLUMNS)}): {list(TARGET_COLUMNS)}")
    print(
        f"Identifier-like columns ({len(identifier_columns)}): "
        f"{identifier_columns}"
    )
    for column in identifier_columns:
        unique_count = data[column].nunique(dropna=False)
        print(
            f"  {column}: {unique_count:,} unique values "
            f"({unique_count / len(data) * 100:.4f}% of rows)"
        )

    all_numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    infinite_counts = pd.Series(0, index=all_numeric_columns, dtype="int64")
    if all_numeric_columns:
        infinite_counts = pd.Series(
            np.isinf(data[all_numeric_columns].to_numpy()).sum(axis=0),
            index=all_numeric_columns,
            dtype="int64",
        )
    print("\nInfinite values in numeric columns:")
    print(infinite_counts.to_string())
    print(f"Total infinite values: {int(infinite_counts.sum()):,}")

    constant_columns = [
        column for column in data.columns if data[column].nunique(dropna=False) <= 1
    ]
    print(f"\nConstant columns: {constant_columns}")

    print("\nBasic descriptive statistics for numeric columns:")
    print(data[all_numeric_columns].describe().transpose().to_string())

    descriptive_categorical_columns = [
        column
        for column in data.columns
        if column in categorical_features or column == "attack_cat"
    ]
    print("\nBasic descriptive statistics for categorical columns:")
    print(
        data[descriptive_categorical_columns]
        .describe()
        .transpose()
        .to_string()
    )

    return data


def compare_datasets(training: pd.DataFrame, testing: pd.DataFrame) -> None:
    """Compare schemas, categorical values, and combined target distributions."""
    print_section("TRAINING VS TESTING COMPARISON")

    schemas_match = training.columns.tolist() == testing.columns.tolist()
    print(f"Column names and order match: {schemas_match}")

    dtype_comparison = pd.DataFrame(
        {
            "training_dtype": training.dtypes.astype(str),
            "testing_dtype": testing.dtypes.astype(str),
        }
    )
    dtype_mismatches = dtype_comparison[
        dtype_comparison["training_dtype"] != dtype_comparison["testing_dtype"]
    ]
    print("\nData type mismatches:")
    print("None" if dtype_mismatches.empty else dtype_mismatches.to_string())

    categorical_columns = training.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    print("\nCategorical values present in only one predefined split:")
    for column in categorical_columns:
        training_values = set(training[column].dropna().astype(str).unique())
        testing_values = set(testing[column].dropna().astype(str).unique())
        training_only = sorted(training_values - testing_values)
        testing_only = sorted(testing_values - training_values)
        print(f"{column}:")
        print(f"  training only: {training_only}")
        print(f"  testing only: {testing_only}")

    combined_label_counts = (
        training["label"].value_counts(dropna=False)
        .add(testing["label"].value_counts(dropna=False), fill_value=0)
        .astype("int64")
        .sort_index()
    )
    combined_attack_counts = (
        training["attack_cat"].value_counts(dropna=False)
        .add(testing["attack_cat"].value_counts(dropna=False), fill_value=0)
        .astype("int64")
        .sort_values(ascending=False)
    )
    combined_rows = len(training) + len(testing)

    print("\nCombined predefined splits - binary label distribution:")
    print(
        pd.DataFrame(
            {
                "count": combined_label_counts,
                "percentage": (
                    combined_label_counts / combined_rows * 100
                ).round(4),
            }
        ).to_string()
    )

    print("\nCombined predefined splits - attack category distribution:")
    print(
        pd.DataFrame(
            {
                "count": combined_attack_counts,
                "percentage": (
                    combined_attack_counts / combined_rows * 100
                ).round(4),
            }
        ).to_string()
    )


def main() -> None:
    """Run the complete read-only dataset inspection."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)

    datasets = [inspect_dataset(path) for path in DATASET_FILES]
    compare_datasets(datasets[0], datasets[1])


if __name__ == "__main__":
    main()
