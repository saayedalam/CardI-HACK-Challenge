# src/prs.py
import pandas as pd
import numpy as np

def rank_snps_on_train(df_train: pd.DataFrame, snp_cols, target: str, method: str = "spearman") -> pd.Series:
    """
    Return a pd.Series of SNP -> correlation with target, sorted by |corr| desc.
    Correlations are computed *only on df_train* to avoid leakage.
    """
    vals = {}
    y = pd.Series(df_train[target].values)
    for c in snp_cols:
        r = pd.Series(df_train[c].values).corr(y, method=method)
        vals[c] = 0.0 if pd.isna(r) else float(r)
    s = pd.Series(vals)
    return s.sort_values(key=lambda v: v.abs(), ascending=False)

def build_prs_per_dataset(df: pd.DataFrame, snp_rank: pd.Series, K: int) -> pd.Series:
    """
    Build a PRS using the top-K SNPs from snp_rank.
    Uses *per-dataset* z-scoring (mean/std computed on the provided df itself),
    then a signed, correlation-weighted sum.
    """
    cols = list(snp_rank.index[:K])
    w    = snp_rank.values[:K]
    Z    = (df[cols] - df[cols].mean()) / (df[cols].std(ddof=0) + 1e-9)
    prs  = (Z.values * w).sum(axis=1)
    return pd.Series(prs, index=df.index, name=f"PRS_K{K}")
