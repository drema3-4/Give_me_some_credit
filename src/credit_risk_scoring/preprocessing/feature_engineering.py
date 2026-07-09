from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, product
from pathlib import Path
import os
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 52
TARGET_COLUMN = "target"

ORIGIN_COLUMNS = [
    "revolving_utilization",
    "age",
    "num_30_59_days_late",
    "debt_ratio",
    "monthly_income",
    "num_open_credit_lines",
    "num_90_days_late",
    "num_real_estate_loans",
    "num_60_89_days_late",
    "num_dependents",
]

LOG_COLUMNS = ORIGIN_COLUMNS.copy()

DISCRETIZATION_COLUMNS = [
    "revolving_utilization",
    "num_30_59_days_late",
    "debt_ratio",
    "monthly_income",
    "num_open_credit_lines",
    "num_90_days_late",
    "num_real_estate_loans",
    "num_60_89_days_late",
    "num_dependents",
]

ALWAYS_FEATURE_COLUMNS = [
    "monthly_income_is_nan",
    "num_dependents_is_nan",
    "revolving_utilization_gt_10",
    "debt_ratio_gt_10",
    "num_30_59_days_late_is_96_or_98",
    "num_60_89_days_late_is_96_or_98",
    "num_90_days_late_is_96_or_98",
]

DISCRETIZATION_BINS = {
    "revolving_utilization": [
        0,
        0.1,
        0.3,
        0.5,
        0.8,
        1.0,
        1.2,
        1.5,
        2.0,
        5.0,
        10.0,
        np.inf,
    ],
    "num_30_59_days_late": [0, 1, 2, 4, 6, 10, 17, 18, 96, 98],
    "debt_ratio": [0, 1, 3, 5, 10, np.inf],
    "monthly_income": [0, 1300, 2029, 3400, 5400, 8250, 11666, 14591.1, 25000, np.inf],
    "num_open_credit_lines": [-np.inf, 0, 2, 3, 5, 8, 11, 15, 18, 24, np.inf],
    "num_90_days_late": [0, 1, 2, 4, 6, 10, 17, 18, 96, 98],
    "num_real_estate_loans": [0, 1, 2, 3, 4, 6, 10, 15, 20, 25, 29, np.inf],
    "num_60_89_days_late": [0, 1, 2, 4, 6, 10, 17, 18, 96, 98],
    "num_dependents": [-1, 0, 1, 2, 3, 4, 5, 7, 10, 20, np.inf],
}

DISCRETIZATION_LABELS = {
    "revolving_utilization": [
        "0-10%",
        "10-30%",
        "30-50%",
        "50-80%",
        "80-100%",
        "100-120%",
        "120-150%",
        "150-200%",
        "200-500%",
        "500-1000%",
        ">1000%",
    ],
    "num_30_59_days_late": [
        "0-1",
        "1-2",
        "2-4",
        "4-6",
        "6-10",
        "10-17",
        "18",
        "19-96",
        "97-98",
    ],
    "debt_ratio": ["0-100%", "100-300%", "300-500%", "500-1000%", ">1000%"],
    "monthly_income": [
        "<1300",
        "1300-2029",
        "2029-3400",
        "3400-5400",
        "5400-8250",
        "8250-11666",
        "11666-14591",
        "14591-25000",
        ">25000",
    ],
    "num_open_credit_lines": ["0", "0-2", "2-3", "3-5", "5-8", "8-11", "11-15", "15-18", "18-24", ">24"],
    "num_90_days_late": [
        "0-1",
        "1-2",
        "2-4",
        "4-6",
        "6-10",
        "10-17",
        "18",
        "19-96",
        "97-98",
    ],
    "num_real_estate_loans": [
        "0-1",
        "1-2",
        "2-3",
        "3-4",
        "4-6",
        "6-10",
        "10-15",
        "15-20",
        "20-25",
        "25-29",
        ">29",
    ],
    "num_60_89_days_late": [
        "0-1",
        "1-2",
        "2-4",
        "4-6",
        "6-10",
        "10-17",
        "18",
        "19-96",
        "97-98",
    ],
    "num_dependents": ["nan", "0", "1", "2", "3", "4", "5-7", "7-10", "10-20", ">20"],
}

MARKER_COLUMNS = [
    "30_59 > 0",
    "60_89 > 0",
    "90 > 0",
    "c_lines = 0",
    "rev_ut [0.5, 10.0]",
    "30_59 > 0 + 60_89 > 0",
    "30_59 > 0 + 60_89 > 0 + d_r [0.5, 3.0]",
    "30_59 > 0 + 90 > 0",
    "60_89 > 0 + 90 > 0",
    "30_59 > 0 + 60_89 > 0 + 90 > 0",
    "rev_ut [0.5, 10.0] + age [18, 52] + 30_59 > 0 + d_r [0.5, 3.0]",
    "age [18, 52] + 60_89 > 0 + d_r [0.5, 3.0] + mon_inc [1300, 5400]",
    "rev_ut [0.5, 10.0] + 60_89 > 0 + d_r [0.5, 3.0]",
    "age [18, 52] + 90 > 0 + d_r [0.5, 3.0]",
    "rev_ut [0.5, 10.0] + 90 > 0 + d_r [0.5, 3.0]",
]

SELECTED_LINEAR_TRANSFORM = {
    "origin": ["age", "num_90_days_late", "num_dependents"],
    "log": ["debt_ratio", "monthly_income", "num_30_59_days_late", "num_60_89_days_late"],
    "discretization": [
        "revolving_utilization",
        "num_open_credit_lines",
        "num_real_estate_loans",
    ],
}

SELECTED_TREE_TRANSFORM = {
    "origin": ORIGIN_COLUMNS.copy(),
    "log": [],
    "discretization": [],
}

SELECTED_LINEAR_MARKERS = [
    "30_59 > 0",
    "60_89 > 0",
    "90 > 0",
    "c_lines = 0",
    "30_59 > 0 + 60_89 > 0",
    "30_59 > 0 + 60_89 > 0 + d_r [0.5, 3.0]",
    "30_59 > 0 + 90 > 0",
    "60_89 > 0 + 90 > 0",
    "rev_ut [0.5, 10.0] + 60_89 > 0 + d_r [0.5, 3.0]",
    "age [18, 52] + 90 > 0 + d_r [0.5, 3.0]",
    "rev_ut [0.5, 10.0] + 90 > 0 + d_r [0.5, 3.0]",
]

SELECTED_TREE_MARKERS = [
    "30_59 > 0",
    "60_89 > 0",
    "90 > 0",
    "rev_ut [0.5, 10.0]",
    "30_59 > 0 + 60_89 > 0 + 90 > 0",
]


FeatureFamily = Literal["linear", "tree"]


def build_feature_space(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create all candidate transformed features and marker columns."""
    feature_data = data.copy()
    raw_data = data.copy()

    feature_data["revolving_utilization_gt_10"] = (feature_data["revolving_utilization"] > 10).astype(int)
    feature_data["revolving_utilization"] = feature_data["revolving_utilization"].clip(upper=10)

    feature_data["debt_ratio_gt_10"] = (feature_data["debt_ratio"] > 10).astype(int)
    feature_data["debt_ratio"] = feature_data["debt_ratio"].clip(upper=10)

    for column in ["num_30_59_days_late", "num_60_89_days_late", "num_90_days_late"]:
        feature_data[f"{column}_is_96_or_98"] = feature_data[column].isin([96, 98]).astype(int)

    for column in LOG_COLUMNS:
        log_values = feature_data[column].astype(float)
        missing_mask = log_values == -1
        feature_data[f"{column}_log"] = np.log1p(log_values.mask(missing_mask, 1.0))
        feature_data.loc[missing_mask, f"{column}_log"] = -np.log(2)

    discretization_columns_new_names: dict[str, list[str]] = {}
    dummy_parts = []

    for column in DISCRETIZATION_COLUMNS:
        binned = pd.cut(
            feature_data[column],
            bins=DISCRETIZATION_BINS[column],
            labels=DISCRETIZATION_LABELS[column],
            right=True,
            include_lowest=True,
        )
        dummies = pd.get_dummies(binned, prefix=f"desc_{column}", dtype=int)
        discretization_columns_new_names[column] = dummies.columns.to_list()
        dummy_parts.append(dummies)

    if dummy_parts:
        feature_data = pd.concat([feature_data, *dummy_parts], axis=1)

    feature_data = add_marker_features(feature_data, raw_data=raw_data)
    return feature_data, discretization_columns_new_names


def add_marker_features(data: pd.DataFrame, raw_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add target-risk markers found during EDA."""
    result = data.copy()
    marker_source = data if raw_data is None else raw_data

    late_30_59 = marker_source["num_30_59_days_late"] > 0
    late_60_89 = marker_source["num_60_89_days_late"] > 0
    late_90 = marker_source["num_90_days_late"] > 0
    rev_util_mid = marker_source["revolving_utilization"].between(0.5, 10.0, inclusive="both")
    age_young_mid = marker_source["age"].between(18, 52, inclusive="both")
    debt_ratio_mid = marker_source["debt_ratio"].between(0.5, 3.0, inclusive="both")
    income_low_mid = marker_source["monthly_income"].between(1300, 5400, inclusive="both")

    result["30_59 > 0"] = late_30_59.astype(int)
    result["60_89 > 0"] = late_60_89.astype(int)
    result["90 > 0"] = late_90.astype(int)
    result["c_lines = 0"] = (marker_source["num_open_credit_lines"] == 0).astype(int)
    result["rev_ut [0.5, 10.0]"] = rev_util_mid.astype(int)
    result["30_59 > 0 + 60_89 > 0"] = (late_30_59 & late_60_89).astype(int)
    result["30_59 > 0 + 60_89 > 0 + d_r [0.5, 3.0]"] = (
        late_30_59 & late_60_89 & debt_ratio_mid
    ).astype(int)
    result["30_59 > 0 + 90 > 0"] = (late_30_59 & late_90).astype(int)
    result["60_89 > 0 + 90 > 0"] = (late_60_89 & late_90).astype(int)
    result["30_59 > 0 + 60_89 > 0 + 90 > 0"] = (late_30_59 & late_60_89 & late_90).astype(int)
    result["rev_ut [0.5, 10.0] + age [18, 52] + 30_59 > 0 + d_r [0.5, 3.0]"] = (
        rev_util_mid & age_young_mid & late_30_59 & debt_ratio_mid
    ).astype(int)
    result["age [18, 52] + 60_89 > 0 + d_r [0.5, 3.0] + mon_inc [1300, 5400]"] = (
        age_young_mid & late_60_89 & debt_ratio_mid & income_low_mid
    ).astype(int)
    result["rev_ut [0.5, 10.0] + 60_89 > 0 + d_r [0.5, 3.0]"] = (
        rev_util_mid & late_60_89 & debt_ratio_mid
    ).astype(int)
    result["age [18, 52] + 90 > 0 + d_r [0.5, 3.0]"] = (
        age_young_mid & late_90 & debt_ratio_mid
    ).astype(int)
    result["rev_ut [0.5, 10.0] + 90 > 0 + d_r [0.5, 3.0]"] = (
        rev_util_mid & late_90 & debt_ratio_mid
    ).astype(int)

    return result


def make_transform_combinations() -> list[dict[str, list[str]]]:
    """Build all valid origin/log/discretization assignments for source features."""
    all_columns = sorted(set(ORIGIN_COLUMNS) | set(LOG_COLUMNS) | set(DISCRETIZATION_COLUMNS))
    available_sources = {}

    for column in all_columns:
        sources = []
        if column in ORIGIN_COLUMNS:
            sources.append("origin")
        if column in LOG_COLUMNS:
            sources.append("log")
        if column in DISCRETIZATION_COLUMNS:
            sources.append("discretization")
        available_sources[column] = sources

    combinations = []
    for assignment in product(*[available_sources[column] for column in all_columns]):
        combination = {"origin": [], "log": [], "discretization": []}
        for column, source in zip(all_columns, assignment):
            combination[source].append(column)
        combinations.append(combination)

    return combinations


def make_marker_combinations(include_empty: bool = True) -> list[tuple[str, ...]]:
    """Build all marker subsets."""
    combinations: list[tuple[str, ...]] = []
    start = 0 if include_empty else 1

    for mask in range(start, 2 ** len(MARKER_COLUMNS)):
        combinations.append(tuple(column for bit, column in enumerate(MARKER_COLUMNS) if mask & (1 << bit)))

    return combinations


def sample_combinations(
    combinations: list,
    n: int | None = None,
    random_state: int = RANDOM_STATE,
) -> list:
    """Return a deterministic subset for smoke runs."""
    if n is None or n >= len(combinations):
        return list(combinations)

    rng = np.random.default_rng(random_state)
    indices = sorted(rng.choice(len(combinations), size=n, replace=False).tolist())
    return [combinations[index] for index in indices]


def feature_columns_from_selection(
    transform_selection: dict[str, list[str]],
    discretization_columns_new_names: dict[str, list[str]],
    marker_columns: Iterable[str] | None = None,
) -> list[str]:
    """Resolve selected source transformations to concrete dataframe columns."""
    marker_columns = list(marker_columns or [])
    origin_features = list(transform_selection.get("origin", []))
    log_features = [f"{column}_log" for column in transform_selection.get("log", [])]
    discretization_features = list(
        chain.from_iterable(
            discretization_columns_new_names[column]
            for column in transform_selection.get("discretization", [])
        )
    )

    columns = [
        *ALWAYS_FEATURE_COLUMNS,
        *origin_features,
        *log_features,
        *discretization_features,
        *marker_columns,
    ]

    return list(dict.fromkeys(columns))


def build_model_dataset(
    data: pd.DataFrame,
    transform_selection: dict[str, list[str]],
    marker_columns: Iterable[str],
) -> pd.DataFrame:
    """Build a final dataset for a chosen model family."""
    feature_data, discretization_columns_new_names = build_feature_space(data)
    feature_columns = feature_columns_from_selection(
        transform_selection,
        discretization_columns_new_names,
        marker_columns=marker_columns,
    )

    output_columns = [TARGET_COLUMN] if TARGET_COLUMN in feature_data.columns else []
    return feature_data[[*output_columns, *feature_columns]].copy()


def build_selected_feature_datasets(
    train_input_path: str | Path,
    test_input_path: str | Path,
    linear_train_output_path: str | Path,
    linear_test_output_path: str | Path,
    tree_train_output_path: str | Path,
    tree_test_output_path: str | Path,
) -> None:
    """Build and save final linear/tree train/test datasets."""
    train_data = pd.read_csv(train_input_path)
    test_data = pd.read_csv(test_input_path)

    outputs = {
        Path(linear_train_output_path): build_model_dataset(
            train_data,
            SELECTED_LINEAR_TRANSFORM,
            SELECTED_LINEAR_MARKERS,
        ),
        Path(linear_test_output_path): build_model_dataset(
            test_data,
            SELECTED_LINEAR_TRANSFORM,
            SELECTED_LINEAR_MARKERS,
        ),
        Path(tree_train_output_path): build_model_dataset(
            train_data,
            SELECTED_TREE_TRANSFORM,
            SELECTED_TREE_MARKERS,
        ),
        Path(tree_test_output_path): build_model_dataset(
            test_data,
            SELECTED_TREE_TRANSFORM,
            SELECTED_TREE_MARKERS,
        ),
    }

    for output_path, dataset in outputs.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)


def evaluate_transform_combinations(
    data: pd.DataFrame,
    combinations: list[dict[str, list[str]]],
    model_family: FeatureFamily,
    n_jobs: int = -1,
    show_progress: bool = True,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Score transform combinations by ROC-AUC, then recall and precision."""
    feature_data, discretization_columns_new_names = build_feature_space(data)
    return _evaluate_feature_sets(
        feature_data=feature_data,
        discretization_columns_new_names=discretization_columns_new_names,
        model_family=model_family,
        transform_combinations=combinations,
        marker_combinations=[()],
        n_jobs=n_jobs,
        show_progress=show_progress,
        random_state=random_state,
    )


def evaluate_marker_combinations(
    data: pd.DataFrame,
    transform_selection: dict[str, list[str]],
    marker_combinations: list[tuple[str, ...]],
    model_family: FeatureFamily,
    n_jobs: int = -1,
    show_progress: bool = True,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Score marker subsets for a fixed source-feature transform selection."""
    feature_data, discretization_columns_new_names = build_feature_space(data)
    return _evaluate_feature_sets(
        feature_data=feature_data,
        discretization_columns_new_names=discretization_columns_new_names,
        model_family=model_family,
        transform_combinations=[transform_selection],
        marker_combinations=marker_combinations,
        n_jobs=n_jobs,
        show_progress=show_progress,
        random_state=random_state,
    )


def sort_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort result rows by the project selection rule."""
    return results.sort_values(
        by=["roc_auc", "recall", "precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _evaluate_feature_sets(
    feature_data: pd.DataFrame,
    discretization_columns_new_names: dict[str, list[str]],
    model_family: FeatureFamily,
    transform_combinations: list[dict[str, list[str]]],
    marker_combinations: list[tuple[str, ...]],
    n_jobs: int,
    show_progress: bool,
    random_state: int,
) -> pd.DataFrame:
    y = feature_data[TARGET_COLUMN].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(feature_data.shape[0]),
        test_size=0.3,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

    tasks = [
        (transform_combination, marker_combination)
        for transform_combination in transform_combinations
        for marker_combination in marker_combinations
    ]

    if show_progress:
        from tqdm.auto import tqdm
    else:
        tqdm = lambda iterable, **_: iterable  # noqa: E731

    worker_count = _resolve_n_jobs(n_jobs)

    if worker_count == 1:
        iterator = (
            _evaluate_one_feature_set(
                feature_data,
                y,
                train_idx,
                test_idx,
                discretization_columns_new_names,
                transform_combination,
                marker_combination,
                model_family,
                random_state,
            )
            for transform_combination, marker_combination in tasks
        )
        results = list(tqdm(iterator, total=len(tasks), desc=f"{model_family} feature sets"))
        return sort_results(pd.DataFrame(results))

    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _evaluate_one_feature_set,
                feature_data,
                y,
                train_idx,
                test_idx,
                discretization_columns_new_names,
                transform_combination,
                marker_combination,
                model_family,
                random_state,
            )
            for transform_combination, marker_combination in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_family} feature sets"):
            results.append(future.result())

    return sort_results(pd.DataFrame(results))


def _evaluate_one_feature_set(
    feature_data: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    discretization_columns_new_names: dict[str, list[str]],
    transform_selection: dict[str, list[str]],
    marker_columns: tuple[str, ...],
    model_family: FeatureFamily,
    random_state: int,
) -> dict:
    feature_columns = feature_columns_from_selection(
        transform_selection,
        discretization_columns_new_names,
        marker_columns=marker_columns,
    )

    X = feature_data[feature_columns]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = _make_model(model_family, random_state=random_state)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "origin_columns": transform_selection.get("origin", []),
        "log_columns": transform_selection.get("log", []),
        "discretization_columns": transform_selection.get("discretization", []),
        "marker_columns": list(marker_columns),
        "final_columns": feature_columns,
        "n_features": len(feature_columns),
    }


def _make_model(model_family: FeatureFamily, random_state: int):
    if model_family == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
        )

    if model_family == "tree":
        return DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
        )

    raise ValueError(f"Unknown model_family: {model_family}")


def _resolve_n_jobs(n_jobs: int) -> int:
    cpu_count = os.cpu_count() or 1

    if n_jobs == -1:
        return max(cpu_count - 1, 1)

    if n_jobs < -1:
        return max(cpu_count + 1 + n_jobs, 1)

    return max(n_jobs, 1)
