"""
Exploratory data analysis helpers: correlation heatmap, per-feature
scatter plots against vacancy rate, and multicollinearity (VIF) check.

Original notebook: 코드정리_분석.ipynb ("EDA, 시각화" / "다중공선성" sections)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import TARGET_COLUMN


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple[int, int] = (20, 15)) -> None:
    """Draw a correlation heatmap over every numeric column in `df`."""
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_feature_scatter(df: pd.DataFrame, columns: list[str], target: str = TARGET_COLUMN) -> None:
    """Grid of scatter plots: each feature in `columns` vs. `target`."""
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols

    plt.figure(figsize=(12, 4 * n_rows))
    for i, col in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.scatterplot(data=df, x=col, y=target)
        plt.title(f"{target} vs. {col}")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for every numeric column in `df`
    (after standardizing and filling missing values with 0).
    """
    numeric_df = df.select_dtypes(include="number").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    vif_df = pd.DataFrame({
        "feature": numeric_df.columns,
        "VIF": [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])],
    })
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
