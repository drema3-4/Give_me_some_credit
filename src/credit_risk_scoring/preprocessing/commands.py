from pathlib import Path
import click

from credit_risk_scoring.preprocessing.common_prepare_train_data import __common_prepare_train_data__


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
    __common_prepare_train_data__(input_path=input_path, output_path=output_path)
