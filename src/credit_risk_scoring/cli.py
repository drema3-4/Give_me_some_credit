import click

from credit_risk_scoring.modeling.commands import train_best_model, train_model
from credit_risk_scoring.preprocessing.commands import (
    build_feature_engineering_datasets,
    common_prepare_test_data,
    common_prepare_train_data,
)


@click.group()
def cli():
    """Credit risk scoring command line tools."""


cli.add_command(common_prepare_train_data)
cli.add_command(common_prepare_test_data)
cli.add_command(build_feature_engineering_datasets)
cli.add_command(train_model)
cli.add_command(train_best_model)


if __name__ == "__main__":
    cli()
