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


def __common_prepare_train_data__(input_path: str | Path, output_path: str | Path) -> None:
    """Common prepare train data: del waste column and rename columns"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    data = pd.read_csv(input_path)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")
    data = data.rename(columns=COLUMNS_MAPPING)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
