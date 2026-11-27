"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def load_brain_data(csv_path: str) -> pd.DataFrame:
    """Загружает brainsize.txt: разделитель — табуляция, пропуски обозначены 'NA'."""
    df = pd.read_csv(csv_path, sep="\t", na_values="NA")
    # В конце файла есть пустые строки — на всякий случай убираем полностью пустые
    return df.dropna(how="all")


def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count.

    Series.corr по умолчанию использует метод Пирсона и учитывает
    только те строки, где оба значения не пропущены.
    """
    result = {}
    for feature in FEATURES:
        result[feature] = float(group[feature].corr(group[TARGET], method="pearson"))
    return result


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = load_brain_data(data.csv_path)

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr = mri_correlations(men)
    women_corr = mri_correlations(women)

    # Ищем признак с максимальным |r| сразу по обеим группам
    strongest_feature = ""
    strongest_value = -1.0
    for correlations in (women_corr, men_corr):
        for feature, value in correlations.items():
            if abs(value) > strongest_value:
                strongest_value = abs(value)
                strongest_feature = feature

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature,
    )