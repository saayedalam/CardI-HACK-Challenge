# src/severity.py
from typing import List, Tuple
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

def train_calibrated_severity(
    df_train: pd.DataFrame,
    features: List[str],
    target_col: str = "OUTCOME SEVERITY",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> CalibratedClassifierCV:
    """
    Trains LightGBM on clinical features and returns an isotonic-calibrated model.
    Uses a fixed CV splitter to make results reproducible.
    """
    base = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=cv)

    X = df_train[features]
    y = df_train[target_col].values
    cal.fit(X, y)
    return cal

def predict_severity_proba(
    model: CalibratedClassifierCV,
    df: pd.DataFrame,
    features: List[str],
) -> pd.Series:
    """
    Returns calibrated P(severity=1) for each row in df.
    """
    proba = model.predict_proba(df[features])[:, 1]
    return pd.Series(proba, index=df.index, name="OUTCOME SEVERITY")
