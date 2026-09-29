"""Data loading and feature engineering."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "Bank Customer Churn Prediction.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "churn"
RAW_NUMERIC = ["credit_score", "age", "tenure", "balance", "products_number",
               "credit_card", "active_member", "estimated_salary"]
RAW_CATEGORICAL = ["country", "gender"]

AGE_BINS = [17, 29, 39, 49, 59, 120]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle file and run basic integrity checks."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'Bank Customer Churn Prediction.csv' from Kaggle "
            "into the dataset/ folder (see dataset/README.md)."
        )
    df = pd.read_csv(path)
    assert df["customer_id"].is_unique, "customer_id should be unique"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    return df.drop(columns=["customer_id"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature construction used by the modeling notebook.

    Groups continuous variables into business-readable bands and adds ratios
    that capture relationship depth with the bank.
    """
    out = df.copy()
    out["age_band"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    out["zero_balance"] = (out["balance"] == 0).astype(int)
    out["balance_to_salary"] = out["balance"] / out["estimated_salary"]
    out["products_group"] = np.select(
        [out["products_number"] == 1, out["products_number"] == 2],
        ["1 product", "2 products"], default="3+ products")
    out["inactive_single_product"] = (
        (out["active_member"] == 0) & (out["products_number"] == 1)).astype(int)
    out["tenure_by_age"] = out["tenure"] / out["age"]
    return out


ENGINEERED_NUMERIC = RAW_NUMERIC + ["zero_balance", "balance_to_salary",
                                    "inactive_single_product", "tenure_by_age"]
ENGINEERED_CATEGORICAL = RAW_CATEGORICAL + ["age_band", "products_group"]
