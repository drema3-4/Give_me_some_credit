import click

from credit_risk_scoring.preprocessing.commands import common_prepare_train_data, common_prepare_test_data


@click.group()
def cli():
    """Credit risk scoring command line tools."""


cli.add_command(common_prepare_train_data)
cli.add_command(common_prepare_test_data)