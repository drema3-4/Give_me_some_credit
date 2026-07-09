from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shlex

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(
    os.getenv("GIVE_ME_SOME_CREDIT_PROJECT_ROOT", "/opt/airflow/give_me_some_credit")
)
DVC_REPRO_CMD = os.getenv("GIVE_ME_SOME_CREDIT_DVC_REPRO_CMD", "dvc repro")

TRAINING_STAGES = [
    "train-logistic-regression",
    "train-decision-tree",
    "train-xgboost",
    "train-catboost",
    "train-random-forest",
]


def dvc_repro_command(stage_name: str) -> str:
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && {DVC_REPRO_CMD} {shlex.quote(stage_name)}"


with DAG(
    dag_id="give_me_some_credit_train_optimal_models",
    description="Prepare processed datasets and train tuned credit risk models.",
    start_date=datetime(2026, 7, 10),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "training", "dvc"],
) as dag:
    build_feature_datasets = BashOperator(
        task_id="build_feature_datasets",
        bash_command=dvc_repro_command("build-feature-engineering-datasets"),
    )

    model_training_tasks = [
        BashOperator(
            task_id=stage_name.replace("-", "_"),
            bash_command=dvc_repro_command(stage_name),
        )
        for stage_name in TRAINING_STAGES
    ]

    build_feature_datasets >> model_training_tasks
