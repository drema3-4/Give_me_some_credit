from pathlib import Path
import click

from credit_risk_scoring.preprocessing.common_prepare_data import common_prepare_data
from credit_risk_scoring.preprocessing.feature_engineering import build_selected_feature_datasets


@click.command("common-prepare-train-data")
@click.option(
    "--input-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/raw/cs-training.csv"),
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/interim/train_prepared.csv"),
    show_default=True,
)
def common_prepare_train_data(input_path: Path, output_path: Path):
    """Common prepare train data"""
    common_prepare_data(input_path=input_path, output_path=output_path)


@click.command("common-prepare-test-data")
@click.option(
    "--input-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/raw/cs-test.csv"),
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/interim/test_prepared.csv"),
    show_default=True,
)
def common_prepare_test_data(input_path: Path, output_path: Path):
    """Common prepare train data"""
    common_prepare_data(input_path=input_path, output_path=output_path)


@click.command("build-feature-engineering-datasets")
@click.option(
    "--train-input-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/interim/train_prepared.csv"),
    show_default=True,
)
@click.option(
    "--test-input-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/interim/test_prepared.csv"),
    show_default=True,
)
@click.option(
    "--linear-train-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/processed/linear/train.csv"),
    show_default=True,
)
@click.option(
    "--linear-test-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/processed/linear/test.csv"),
    show_default=True,
)
@click.option(
    "--tree-train-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/processed/tree/train.csv"),
    show_default=True,
)
@click.option(
    "--tree-test-output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/processed/tree/test.csv"),
    show_default=True,
)
def build_feature_engineering_datasets(
    train_input_path: Path,
    test_input_path: Path,
    linear_train_output_path: Path,
    linear_test_output_path: Path,
    tree_train_output_path: Path,
    tree_test_output_path: Path,
):
    """Build selected feature-engineered datasets."""
    build_selected_feature_datasets(
        train_input_path=train_input_path,
        test_input_path=test_input_path,
        linear_train_output_path=linear_train_output_path,
        linear_test_output_path=linear_test_output_path,
        tree_train_output_path=tree_train_output_path,
        tree_test_output_path=tree_test_output_path,
    )
