from pathlib import Path

import click

from credit_risk_scoring.modeling.train_models import (
    MODEL_CONFIGS,
    RANDOM_STATE,
    train_fixed_and_save_model,
    train_and_save_model,
)


@click.command("train-model")
@click.option(
    "--model-type",
    type=click.Choice(sorted(MODEL_CONFIGS.keys())),
    required=True,
    help="Model family to tune and train.",
)
@click.option(
    "--train-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Processed train dataset path.",
)
@click.option(
    "--model-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for the fitted model artifact.",
)
@click.option(
    "--metrics-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for validation metrics JSON.",
)
@click.option(
    "--params-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for best hyperparameters JSON.",
)
@click.option(
    "--cv-results-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for cross-validation results CSV.",
)
@click.option(
    "--search-strategy",
    type=click.Choice(["grid", "random"]),
    default="grid",
    show_default=True,
)
@click.option("--scoring", default="roc_auc", show_default=True)
@click.option("--cv", type=click.IntRange(min=2), default=5, show_default=True)
@click.option("--n-jobs", type=int, default=-1, show_default=True)
@click.option("--n-iter", type=click.IntRange(min=1), default=25, show_default=True)
@click.option("--test-size", type=click.FloatRange(min=0.05, max=0.5), default=0.2, show_default=True)
@click.option("--random-state", type=int, default=RANDOM_STATE, show_default=True)
def train_model(
    model_type: str,
    train_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    params_output_path: Path,
    cv_results_output_path: Path | None,
    search_strategy: str,
    scoring: str,
    cv: int,
    n_jobs: int,
    n_iter: int,
    test_size: float,
    random_state: int,
):
    """Tune hyperparameters, train the best model, and save artifacts."""
    result = train_and_save_model(
        train_path=train_path,
        model_type=model_type,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        params_output_path=params_output_path,
        cv_results_output_path=cv_results_output_path,
        search_strategy=search_strategy,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        n_iter=n_iter,
        test_size=test_size,
        random_state=random_state,
    )

    validation = result["metrics"]["validation"]
    click.echo(
        f"{model_type}: best_cv_score={result['metrics']['best_cv_score']:.5f}, "
        f"validation_roc_auc={validation['roc_auc']:.5f}, "
        f"validation_recall={validation['recall']:.5f}, "
        f"validation_precision={validation['precision']:.5f}"
    )


@click.command("train-best-model")
@click.option(
    "--model-type",
    type=click.Choice(sorted(MODEL_CONFIGS.keys())),
    required=True,
    help="Model family to train with selected best hyperparameters.",
)
@click.option(
    "--train-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Processed train dataset path.",
)
@click.option(
    "--model-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for the fitted model artifact.",
)
@click.option(
    "--metrics-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for validation metrics JSON.",
)
@click.option(
    "--params-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for selected hyperparameters JSON.",
)
@click.option("--test-size", type=click.FloatRange(min=0.05, max=0.5), default=0.2, show_default=True)
@click.option("--random-state", type=int, default=RANDOM_STATE, show_default=True)
def train_best_model(
    model_type: str,
    train_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    params_output_path: Path,
    test_size: float,
    random_state: int,
):
    """Train with fixed selected hyperparameters and save artifacts."""
    result = train_fixed_and_save_model(
        train_path=train_path,
        model_type=model_type,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        params_output_path=params_output_path,
        test_size=test_size,
        random_state=random_state,
    )

    validation = result["metrics"]["validation"]
    selection = result["metrics"]["selection_metrics"]
    click.echo(
        f"{model_type}: selected_cv_roc_auc={selection['best_cv_roc_auc']:.5f}, "
        f"validation_roc_auc={validation['roc_auc']:.5f}, "
        f"validation_recall={validation['recall']:.5f}, "
        f"validation_precision={validation['precision']:.5f}"
    )
