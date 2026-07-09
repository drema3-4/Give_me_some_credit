from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 52
TARGET_COLUMN = "target"

ModelType = Literal[
    "logistic_regression",
    "decision_tree",
    "xgboost",
    "catboost",
    "random_forest",
]
SearchStrategy = Literal["grid", "random"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration needed to tune and train one model family."""

    dataset_family: Literal["linear", "tree"]
    estimator: Any
    param_grid: dict[str, list[Any]]


class SafeXGBoostFeatureNames(BaseEstimator, TransformerMixin):
    """Rename dataframe columns to names accepted by XGBoost."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if not isinstance(X, pd.DataFrame):
            self.safe_feature_names_ = None
            return self

        self.safe_feature_names_ = _make_safe_feature_names(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or self.safe_feature_names_ is None:
            return X

        result = X.copy()
        result.columns = self.safe_feature_names_
        return result


MODEL_CONFIGS: dict[ModelType, ModelConfig] = {
    "logistic_regression": ModelConfig(
        dataset_family="linear",
        estimator=Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        param_grid={
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
    ),
    "decision_tree": ModelConfig(
        dataset_family="tree",
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [20, 50, 100, 200],
            "min_samples_split": [50, 100, 200],
        },
    ),
    "xgboost": ModelConfig(
        dataset_family="tree",
        estimator=Pipeline(
            steps=[
                ("safe_feature_names", SafeXGBoostFeatureNames()),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ],
        ),
        param_grid={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
        },
    ),
    "catboost": ModelConfig(
        dataset_family="tree",
        estimator=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            allow_writing_files=False,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        param_grid={
            "iterations": [300, 500, 700],
            "depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [3.0, 7.0, 10.0],
        },
    ),
    "random_forest": ModelConfig(
        dataset_family="tree",
        estimator=RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [300, 500],
            "max_depth": [5, 8, 12, None],
            "min_samples_leaf": [20, 50, 100],
            "max_features": ["sqrt", 0.6],
        },
    ),
}

BEST_MODEL_PARAMS: dict[ModelType, dict[str, Any]] = {
    "logistic_regression": {
        "model__C": 5.0,
        "model__solver": "liblinear",
    },
    "decision_tree": {
        "criterion": "entropy",
        "max_depth": 7,
        "min_samples_leaf": 200,
        "min_samples_split": 50,
    },
    "xgboost": {
        "model__colsample_bytree": 0.8,
        "model__learning_rate": 0.03,
        "model__max_depth": 4,
        "model__n_estimators": 400,
        "model__reg_lambda": 10.0,
        "model__subsample": 1.0,
    },
    "catboost": {
        "depth": 6,
        "iterations": 700,
        "l2_leaf_reg": 10.0,
        "learning_rate": 0.03,
    },
    "random_forest": {
        "max_depth": 12,
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "n_estimators": 500,
    },
}

BEST_MODEL_SELECTION_METRICS: dict[ModelType, dict[str, float]] = {
    "logistic_regression": {
        "best_cv_roc_auc": 0.8610401191412933,
        "validation_roc_auc": 0.8592388741979788,
        "validation_average_precision": 0.388077588560066,
        "validation_precision": 0.22451127819548872,
        "validation_recall": 0.7457542457542458,
        "validation_f1": 0.3451225150254276,
    },
    "decision_tree": {
        "best_cv_roc_auc": 0.8540568562786408,
        "validation_roc_auc": 0.8485348236209191,
        "validation_average_precision": 0.36803801021016347,
        "validation_precision": 0.1994878361075544,
        "validation_recall": 0.7782217782217782,
        "validation_f1": 0.317570322054627,
    },
    "xgboost": {
        "best_cv_roc_auc": 0.8668516036039463,
        "validation_roc_auc": 0.8637941194979508,
        "validation_average_precision": 0.39863251714590314,
        "validation_precision": 0.21431520991052994,
        "validation_recall": 0.7777222777222778,
        "validation_f1": 0.3360310780187763,
    },
    "catboost": {
        "best_cv_roc_auc": 0.867065132238911,
        "validation_roc_auc": 0.8634975836331842,
        "validation_average_precision": 0.4003103230840541,
        "validation_precision": 0.21794871794871795,
        "validation_recall": 0.7642357642357642,
        "validation_f1": 0.3391709155397916,
    },
    "random_forest": {
        "best_cv_roc_auc": 0.864675717870858,
        "validation_roc_auc": 0.8610582271796219,
        "validation_average_precision": 0.39211186075616344,
        "validation_precision": 0.2291988864831426,
        "validation_recall": 0.7402597402597403,
        "validation_f1": 0.3500236183278224,
    },
}


def load_training_data(train_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed training dataset and split target from features."""
    data = pd.read_csv(train_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' was not found in {train_path}")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return X, y


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: ModelType,
    search_strategy: SearchStrategy = "grid",
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    n_iter: int = 25,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV | RandomizedSearchCV:
    """Tune model hyperparameters with stratified cross-validation."""
    config = MODEL_CONFIGS[model_type]
    estimator = clone(config.estimator)
    param_grid = dict(config.param_grid)

    if model_type == "xgboost":
        estimator.set_params(model__scale_pos_weight=_scale_pos_weight(y))

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    if search_strategy == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            verbose=1,
        )
    elif search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            random_state=random_state,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown search_strategy: {search_strategy}")

    search.fit(X, y)
    return search


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a fitted binary classifier."""
    y_score = _predict_positive_score(model, X)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, y_score)),
        "average_precision": float(average_precision_score(y, y_score)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "threshold": float(threshold),
    }


def train_tuned_model(
    train_path: str | Path,
    model_type: ModelType,
    search_strategy: SearchStrategy = "grid",
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    n_iter: int = 25,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Tune on a train split, refit the best model on all data, and return artifacts."""
    X, y = load_training_data(train_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    search = tune_model(
        X_train,
        y_train,
        model_type=model_type,
        search_strategy=search_strategy,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        n_iter=n_iter,
        random_state=random_state,
    )

    validation_metrics = evaluate_model(search.best_estimator_, X_valid, y_valid)
    train_metrics = evaluate_model(search.best_estimator_, X_train, y_train)

    final_model = clone(search.best_estimator_)
    if model_type == "xgboost":
        final_model.set_params(model__scale_pos_weight=_scale_pos_weight(y))

    final_model.fit(X, y)

    metrics = {
        "model_type": model_type,
        "dataset_family": MODEL_CONFIGS[model_type].dataset_family,
        "train_path": str(train_path),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_rate": float(y.mean()),
        "search_strategy": search_strategy,
        "scoring": scoring,
        "cv": int(cv),
        "best_cv_score": float(search.best_score_),
        "best_params": _json_safe(search.best_params_),
        "train": train_metrics,
        "validation": validation_metrics,
    }

    return {
        "model": final_model,
        "metrics": metrics,
        "best_params": search.best_params_,
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def train_fixed_model(
    train_path: str | Path,
    model_type: ModelType,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train one model family with fixed selected hyperparameters."""
    X, y = load_training_data(train_path)
    params = dict(BEST_MODEL_PARAMS[model_type] if params is None else params)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_model = _make_configured_estimator(model_type, params, y_train)
    validation_model.fit(X_train, y_train)

    validation_metrics = evaluate_model(validation_model, X_valid, y_valid)
    train_metrics = evaluate_model(validation_model, X_train, y_train)

    final_model = _make_configured_estimator(model_type, params, y)
    final_model.fit(X, y)

    metrics = {
        "model_type": model_type,
        "dataset_family": MODEL_CONFIGS[model_type].dataset_family,
        "training_mode": "fixed_best_params",
        "train_path": str(train_path),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_rate": float(y.mean()),
        "params": _json_safe(params),
        "selection_metrics": BEST_MODEL_SELECTION_METRICS[model_type],
        "train": train_metrics,
        "validation": validation_metrics,
    }

    return {
        "model": final_model,
        "metrics": metrics,
        "best_params": params,
        "cv_results": pd.DataFrame(),
    }


def save_training_artifacts(
    training_result: dict[str, Any],
    model_output_path: str | Path,
    metrics_output_path: str | Path,
    params_output_path: str | Path,
    cv_results_output_path: str | Path | None = None,
) -> None:
    """Persist a trained model and tuning reports."""
    model_output_path = Path(model_output_path)
    metrics_output_path = Path(metrics_output_path)
    params_output_path = Path(params_output_path)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    params_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result["model"], model_output_path)

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(training_result["metrics"], file, ensure_ascii=False, indent=2)

    with params_output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(training_result["best_params"]), file, ensure_ascii=False, indent=2)

    if cv_results_output_path is not None:
        cv_results_output_path = Path(cv_results_output_path)
        cv_results_output_path.parent.mkdir(parents=True, exist_ok=True)
        training_result["cv_results"].to_csv(cv_results_output_path, index=False)


def train_and_save_model(
    train_path: str | Path,
    model_type: ModelType,
    model_output_path: str | Path,
    metrics_output_path: str | Path,
    params_output_path: str | Path,
    cv_results_output_path: str | Path | None = None,
    search_strategy: SearchStrategy = "grid",
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    n_iter: int = 25,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full tune/train/save flow."""
    result = train_tuned_model(
        train_path=train_path,
        model_type=model_type,
        search_strategy=search_strategy,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        n_iter=n_iter,
        test_size=test_size,
        random_state=random_state,
    )
    save_training_artifacts(
        result,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        params_output_path=params_output_path,
        cv_results_output_path=cv_results_output_path,
    )
    return result


def train_fixed_and_save_model(
    train_path: str | Path,
    model_type: ModelType,
    model_output_path: str | Path,
    metrics_output_path: str | Path,
    params_output_path: str | Path,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train and save a model with fixed selected hyperparameters."""
    result = train_fixed_model(
        train_path=train_path,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
    )
    save_training_artifacts(
        result,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        params_output_path=params_output_path,
    )
    return result


def default_train_path(project_root: str | Path, model_type: ModelType) -> Path:
    """Return the processed dataset path for a model family."""
    dataset_family = MODEL_CONFIGS[model_type].dataset_family
    return Path(project_root) / "data" / "processed" / dataset_family / "train.csv"


def _predict_positive_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1 / (1 + np.exp(-scores))

    raise TypeError(f"Model {type(model).__name__} does not expose probabilities or decision scores.")


def _make_configured_estimator(
    model_type: ModelType,
    params: dict[str, Any],
    y: pd.Series,
) -> Any:
    estimator = clone(MODEL_CONFIGS[model_type].estimator)
    estimator.set_params(**params)

    if model_type == "xgboost":
        estimator.set_params(model__scale_pos_weight=_scale_pos_weight(y))

    return estimator


def _scale_pos_weight(y: pd.Series) -> float:
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def _make_safe_feature_names(columns: pd.Index) -> list[str]:
    safe_names = []
    used_names = set()

    for index, column in enumerate(columns):
        name = re.sub(r"[\[\]<>]", "_", str(column))
        name = re.sub(r"\s+", "_", name).strip("_")
        name = name or f"feature_{index}"

        candidate = name
        counter = 1
        while candidate in used_names:
            counter += 1
            candidate = f"{name}_{counter}"

        used_names.add(candidate)
        safe_names.append(candidate)

    return safe_names


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    return value
