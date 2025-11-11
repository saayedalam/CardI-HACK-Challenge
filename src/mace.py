# src/mace.py
from typing import List, Dict, Sequence
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

def train_mace_model(
    df_train: pd.DataFrame,
    features: List[str],
    target_col: str = "OUTCOME MACE",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    class_weight: Dict[int, float] = None,
    seed: int = 42,
) -> LGBMClassifier:
    """
    Train a single LightGBM multiclass model for MACE.
    """
    if class_weight is None:
        class_weight = {0: 1, 1: 3, 2: 4}

    X = df_train[features]
    y = df_train[target_col].values

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=seed,
    )
    model.fit(X, y)
    return model

def predict_mace_labels(
    model: LGBMClassifier,
    df: pd.DataFrame,
    features: List[str],
) -> pd.Series:
    """
    Predict class labels (0/1/2) for MACE.
    """
    preds = model.predict(df[features])
    return pd.Series(preds, index=df.index, name="OUTCOME MACE")

def train_mace_bag(
    df_train: pd.DataFrame,
    features: List[str],
    seeds: Sequence[int] = (11, 19, 23, 37, 42),
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    class_weight: Dict[int, float] = None,
) -> Sequence[LGBMClassifier]:
    """
    Train multiple MACE models with different seeds for majority-vote bagging.
    """
    if class_weight is None:
        class_weight = {0: 1, 1: 3, 2: 4}

    models = []
    for s in seeds:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            class_weight=class_weight,
            random_state=s,
        )
        model.fit(df_train[features], df_train["OUTCOME MACE"].values)
        models.append(model)
    return models

def predict_mace_bag_labels(
    models: Sequence[LGBMClassifier],
    df: pd.DataFrame,
    features: List[str],
) -> pd.Series:
    """
    Majority vote across a list of trained models.
    Ties broken by the last model in the list.
    """
    pred_matrix = np.vstack([m.predict(df[features]) for m in models])  # (n_models, n_samples)

    # majority vote, tie-break with last model
    out = []
    last = pred_matrix[-1]
    for j in range(pred_matrix.shape[1]):
        col = pred_matrix[:, j]
        vals, counts = np.unique(col, return_counts=True)
        top = vals[np.argmax(counts)]
        # if a perfect tie across all models (rare), fall back to last
        if (counts == counts.max()).sum() > 1:
            top = last[j]
        out.append(top)

    return pd.Series(np.array(out, dtype=int), index=df.index, name="OUTCOME MACE")
