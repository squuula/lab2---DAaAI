"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def load_titanic(csv_path: str) -> pd.DataFrame:
    """Загружает titanic.csv в DataFrame."""
    return pd.read_csv(csv_path)


def count_missing(df: pd.DataFrame) -> dict[str, int]:
    """Количество пропусков в каждом столбце."""
    missing = df.isna().sum()
    return {str(column): int(count) for column, count in missing.items()}


def group_by_pclass(df: pd.DataFrame, column: str) -> dict[int, float]:
    """Среднее значение столбца column для каждого класса Pclass.

    mean() в pandas пропускает NaN, поэтому пассажиры без возраста
    просто не участвуют в расчёте (а не считаются как возраст 0).
    """
    means = df.groupby("Pclass")[column].mean()
    return {int(pclass): float(value) for pclass, value in means.items()}


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = load_titanic(data.csv_path)

    # Сравнение NaN > 30 даёт False, поэтому пропуски сюда не попадут
    adults_over_30 = int((df["Age"] > 30).sum())

    # Survived принимает значения 0/1, значит среднее = доля выживших
    mean_age = group_by_pclass(df, "Age")
    survival_rate = group_by_pclass(df, "Survived")

    top_fares = df["Fare"].dropna().sort_values(ascending=False).head(5)
    highest_fares = [float(fare) for fare in top_fares]

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=count_missing(df),
        adults_over_30_count=adults_over_30,
        mean_age_by_pclass=mean_age,
        survival_rate_by_pclass=survival_rate,
        highest_fares=highest_fares,
    )
