from pathlib import Path
import pandas as pd


COLUMNS_MAPPING = {
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "num_30_59_days_late",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "num_open_credit_lines",
    "NumberOfTimes90DaysLate": "num_90_days_late",
    "NumberRealEstateLoansOrLines": "num_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "num_60_89_days_late",
    "NumberOfDependents": "num_dependents"
}


def common_prepare_data(input_path: str | Path, output_path: str | Path) -> None:
    """Common prepare train data: del waste column and rename columns"""
    #########################################################################
    # настройка
    #########################################################################
    input_path = Path(input_path)
    output_path = Path(output_path)
    #########################################################################
    # загрузка данных и удобства
    #########################################################################
    data = pd.read_csv(input_path)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")
    data = data.rename(columns=COLUMNS_MAPPING)
    #########################################################################
    # общее качество данных
    #########################################################################
    data = data.drop_duplicates()

    data = data.query("revolving_utilization >= 0")
    data = data.query("age > 0")
    data = data.query("num_30_59_days_late >= 0")
    data = data.query("debt_ratio >= 0")
    data = data.query("(monthly_income.isna()) or (monthly_income >= 0)")
    data = data.query("num_open_credit_lines >= 0")
    data = data.query("num_90_days_late >= 0")
    data = data.query("num_real_estate_loans >= 0")
    data = data.query("num_60_89_days_late >= 0")
    data = data.query("(monthly_income.isna()) or (monthly_income >= 0)")    
    #########################################################################
    # обработка признаков
    #########################################################################
    # Так как в левой части признак имеет положительную линейную связь с таргетом,
    # то чем меньше значение до nan, тем меньше риск дефолта
    # Так как признак положителен, то значение -1 подойдёт
    data["monthly_income_is_nan"] = data["monthly_income"].isna().astype(int)
    data["monthly_income"] = data["monthly_income"].fillna(-1)

    # Так как в левой части признак имеет положительную линейную связь с таргетом,
    # то чем меньше значение до nan, тем меньше риск дефолта
    # Так как признак положителен, то значение -1 подойдёт
    data["num_dependents_is_nan"] = data["num_dependents"].isna().astype(int)
    data["num_dependents"] = data["num_dependents"].fillna(-1)
    #########################################################################
    # сохранение
    #########################################################################
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
