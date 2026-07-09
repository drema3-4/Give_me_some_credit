# Best Model Selection Results

Подбор выполнен на `data/processed/linear/train.csv` для логистической регрессии и на `data/processed/tree/train.csv` для остальных моделей.

Критерий подбора: `roc_auc` на 5-fold stratified CV. Дополнительно качество проверено на holdout-валидации `test_size=0.2`, `random_state=52`.

## Итоговая таблица

| Model | Dataset | Best CV ROC-AUC | Validation ROC-AUC | Validation Average Precision | Validation Precision | Validation Recall | Validation F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | linear | 0.861040 | 0.859239 | 0.388078 | 0.224511 | 0.745754 | 0.345123 |
| decision_tree | tree | 0.854057 | 0.848535 | 0.368038 | 0.199488 | 0.778222 | 0.317570 |
| xgboost | tree | 0.866852 | 0.863794 | 0.398633 | 0.214315 | 0.777722 | 0.336031 |
| catboost | tree | 0.867065 | 0.863498 | 0.400310 | 0.217949 | 0.764236 | 0.339171 |
| random_forest | tree | 0.864676 | 0.861058 | 0.392112 | 0.229199 | 0.740260 | 0.350024 |

Лучшая модель по holdout ROC-AUC: `xgboost`, `validation_roc_auc = 0.863794`.

Лучшая модель по CV ROC-AUC: `catboost`, `best_cv_roc_auc = 0.867065`.

## Лучшие параметры

### logistic_regression

```json
{
  "model__C": 5.0,
  "model__solver": "liblinear"
}
```

### decision_tree

```json
{
  "criterion": "entropy",
  "max_depth": 7,
  "min_samples_leaf": 200,
  "min_samples_split": 50
}
```

### xgboost

```json
{
  "model__colsample_bytree": 0.8,
  "model__learning_rate": 0.03,
  "model__max_depth": 4,
  "model__n_estimators": 400,
  "model__reg_lambda": 10.0,
  "model__subsample": 1.0
}
```

### catboost

```json
{
  "depth": 6,
  "iterations": 700,
  "l2_leaf_reg": 10.0,
  "learning_rate": 0.03
}
```

### random_forest

```json
{
  "max_depth": 12,
  "max_features": "sqrt",
  "min_samples_leaf": 20,
  "n_estimators": 500
}
```

## Артефакты подбора

- `reports/modeling/model_selection_results.json` - полные метрики и параметры по каждой модели.
- `reports/modeling/model_selection_results.csv` - компактная таблица для просмотра.
- `reports/modeling/*_tuning_cv_results.csv` - подробные CV-результаты по сеткам подбора.
