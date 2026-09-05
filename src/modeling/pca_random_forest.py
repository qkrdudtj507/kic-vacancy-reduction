"""
Vacancy-rate prediction model: StandardScaler -> PCA -> RandomForestRegressor.

This was selected as the final model after comparing GWR, K-means
clustering, plain multiple regression, and logistic regression - see
the project README for the comparison table. PCA removes the heavy
multicollinearity between raw features (VIF > 10 for many columns)
while RandomForest captures non-linear relationships that the linear
models could not.

Original notebook: 코드정리_분석.ipynb ("PCA + RandomForest" section),
공공공_상관관계_다중회귀_세희추가.ipynb
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_FEATURE_COLUMNS,
    PCA_N_COMPONENTS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class VacancyRatePredictor:
    """Fit once on the Seongnam training data, then reuse the same
    fitted scaler / PCA / RandomForest to score any new region (e.g.
    the Hanam-Gyosan candidate grid) on a consistent feature space.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_components: int = PCA_N_COMPONENTS,
        n_estimators: int = 100,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.test_metrics_: dict[str, float] | None = None

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.feature_columns if c in df.columns]
        return df[available].fillna(0)

    def fit(self, df: pd.DataFrame, target_column: str = TARGET_COLUMN, test_size: float = TEST_SIZE) -> "VacancyRatePredictor":
        """Fit the scaler, PCA, and RandomForest on `df`, holding out
        `test_size` for evaluation. Populates `self.test_metrics_`.
        """
        X = self._select_features(df)
        y = df[target_column]

        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca_df, y, test_size=test_size, random_state=RANDOM_STATE
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.test_metrics_ = {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        return self

    def pca_loadings(self) -> pd.DataFrame:
        """Variable loadings for every principal component (rows = PCs)."""
        return pd.DataFrame(
            self.pca.components_,
            columns=self.feature_columns[: self.pca.components_.shape[1]],
            index=[f"PC{i + 1}" for i in range(self.pca.n_components_)],
        )

    def feature_importance(self) -> pd.DataFrame:
        """RandomForest feature importance, one row per principal component."""
        importance = pd.DataFrame({
            "component": [f"PC{i + 1}" for i in range(self.pca.n_components_)],
            "importance": self.model.feature_importances_,
        })
        return importance.sort_values("importance", ascending=False).reset_index(drop=True)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict vacancy rate for new rows using the already-fitted
        scaler / PCA / RandomForest.
        """
        X = self._select_features(df)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)
