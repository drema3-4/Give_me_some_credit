# Give Me Some Credit

Проект по кредитному скорингу на данных соревнования Kaggle Give Me Some Credit. Цель - предсказать риск серьезной просрочки по заемщику и собрать воспроизводимый ML-пайплайн от сырых данных до обученных моделей.

Основной таргет: `SeriousDlqin2yrs`, в проектном коде переименован в `target`.

## Результаты проекта

В проекте выполнены EDA, очистка данных, feature engineering, подбор признаков под линейные и деревообразные модели, подбор гиперпараметров и финальное обучение моделей.

| Модель | Датасет | CV ROC-AUC | Validation ROC-AUC | Validation Recall | Validation Precision |
|---|---|---:|---:|---:|---:|
| logistic_regression | linear | 0.861040 | 0.859239 | 0.745754 | 0.224511 |
| decision_tree | tree | 0.854057 | 0.848535 | 0.778222 | 0.199488 |
| xgboost | tree | 0.866852 | 0.863794 | 0.777722 | 0.214315 |
| catboost | tree | 0.867065 | 0.863498 | 0.764236 | 0.217949 |
| random_forest | tree | 0.864676 | 0.861058 | 0.740260 | 0.229199 |

Лучший результат по holdout ROC-AUC показал `xgboost`: `0.863794`.

Лучший результат по CV ROC-AUC показал `catboost`: `0.867065`.

Итоговые артефакты:

- обученные модели лежат в `models/*.joblib`;
- метрики и лучшие параметры лежат в `reports/modeling`;
- воспроизводимый пайплайн описан в `dvc.yaml`;
- Airflow DAG для запуска финального обучения лежит в `airflow/dags/train_optimal_models.py`.

## Технологии проекта

- Python 3.12 - основной язык проекта.
- Poetry - управление зависимостями и запуск CLI.
- pandas, numpy - обработка данных.
- scikit-learn - preprocessing, baseline-модели, метрики, подбор гиперпараметров.
- XGBoost, CatBoost - градиентный бустинг.
- joblib - сохранение обученных моделей.
- DVC, dvc-gdrive - версионирование данных, моделей, метрик и запуск пайплайна.
- Airflow - оркестрация DVC-этапов обучения через DAG.
- MLflow - в текущем состоянии репозитория активный tracking не подключен; статус и рекомендации зафиксированы в отдельной документации.
- Jupyter Notebook - исследовательские этапы EDA, feature engineering и моделирования.

## Документация

Документы лежат в папке `docs`:

- `docs/01_eda_data_summary.pdf` - краткий отчет по EDA.
- `docs/02_feature_engineering_summary.pdf` - краткий отчет по feature engineering.
- `docs/03_train_models_summary.pdf` - краткий отчет по обучению моделей.
- `docs/dvc_project_documentation.pdf` - как DVC устроен в проекте.
- `docs/mlflow_project_documentation.pdf` - текущий статус MLflow и как фиксируются эксперименты.
- `docs/airflow_project_documentation.pdf` - как устроен Airflow DAG.

## Структура проекта

```text
.
|-- airflow/
|   `-- dags/
|       `-- train_optimal_models.py
|-- data/
|   |-- raw/                 # исходные данные, tracked через DVC
|   |-- interim/             # очищенные train/test таблицы
|   `-- processed/           # финальные датасеты для linear и tree моделей
|-- docs/                    # PDF-документация проекта
|-- models/                  # обученные joblib-модели
|-- notebooks/               # исследовательские ноутбуки
|-- reports/
|   `-- modeling/            # метрики, параметры, результаты подбора
|-- src/
|   `-- credit_risk_scoring/
|       |-- preprocessing/   # очистка данных и feature engineering
|       |-- modeling/        # обучение, подбор, сохранение моделей
|       `-- cli.py           # CLI credit-risk
|-- dvc.yaml                 # DVC-пайплайн
|-- dvc.lock                 # зафиксированное состояние DVC-пайплайна
|-- pyproject.toml           # зависимости и entrypoint
`-- poetry.lock
```

## Как запускать

### Установка

```powershell
python -m pip install poetry
poetry install
```

### Получить данные и артефакты из DVC

```powershell
poetry run dvc pull
```

Если данные уже лежат локально в `data/raw`, можно сразу запускать пайплайн.

### Запустить полный DVC-пайплайн

```powershell
poetry run dvc repro
```

### Запустить отдельный DVC-этап

```powershell
poetry run dvc repro build-feature-engineering-datasets
poetry run dvc repro train-xgboost
```

### Посмотреть метрики DVC

```powershell
poetry run dvc metrics show
```

### Запустить команды проекта напрямую

```powershell
poetry run credit-risk common-prepare-train-data
poetry run credit-risk common-prepare-test-data
poetry run credit-risk build-feature-engineering-datasets
```

Пример обучения одной модели:

```powershell
poetry run credit-risk train-best-model `
  --model-type xgboost `
  --train-path data/processed/tree/train.csv `
  --model-output-path models/xgboost.joblib `
  --metrics-output-path reports/modeling/xgboost_metrics.json `
  --params-output-path reports/modeling/xgboost_best_params.json
```

### Запустить Airflow DAG

DAG находится в `airflow/dags/train_optimal_models.py` и рассчитан на ручной запуск.

В окружении Airflow репозиторий должен быть доступен по пути из переменной `GIVE_ME_SOME_CREDIT_PROJECT_ROOT`. По умолчанию используется `/opt/airflow/give_me_some_credit`.

Пример переменных для Airflow:

```bash
export GIVE_ME_SOME_CREDIT_PROJECT_ROOT=/opt/airflow/give_me_some_credit
export GIVE_ME_SOME_CREDIT_DVC_REPRO_CMD="poetry run dvc repro"
```

После этого в Airflow UI нужно запустить DAG `give_me_some_credit_train_optimal_models`.
