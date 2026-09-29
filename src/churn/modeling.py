"""Model definitions, cross-validation and business evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "f1": "f1"}


def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def base_models():
    """The six base classifiers compared in the original study.

    Models run single-threaded; parallelism happens across CV folds instead.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                n_jobs=1, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "SVM (RBF)": SVC(C=1.0, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 n_jobs=1, random_state=SEED),
    }


def ensemble_models():
    """Stacking and soft-voting ensembles built from the strongest base learners."""
    m = base_models()
    members = [("rf", m["Random Forest"]), ("xgb", m["XGBoost"]), ("lr", m["Logistic Regression"])]
    return {
        "Stacking (RF + XGB + LR)": StackingClassifier(
            estimators=members, final_estimator=LogisticRegression(max_iter=2000),
            cv=5, stack_method="predict_proba", n_jobs=1),
        "Soft Voting (RF + XGB + LR)": VotingClassifier(estimators=members, voting="soft", n_jobs=1),
    }


def make_pipeline(model, numeric, categorical):
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", model)])


def cross_validate_models(models, X, y, numeric, categorical, folds=10):
    """10-fold stratified CV; returns mean and std for each metric."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        res = cross_validate(make_pipeline(model, numeric, categorical), X, y,
                             cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = res[f"test_{metric}"].mean()
            row[f"{metric}_std"] = res[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def gains_table(y_true, y_score, bins=10):
    """Decile gains/lift table: how many churners are captured by targeting the riskiest customers."""
    d = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    d["decile"] = np.arange(len(d)) * bins // len(d) + 1
    g = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["cum_customers_pct"] = g["customers"].cumsum() / g["customers"].sum()
    g["cum_churners_captured_pct"] = g["churners"].cumsum() / g["churners"].sum()
    g["lift"] = g["churn_rate"] / d["y"].mean()
    return g


def retention_campaign_profit(y_true, y_score, customer_value, offer_cost, save_rate):
    """Expected campaign profit when targeting the top-k% riskiest customers.

    profit(k) = churners_in_top_k * save_rate * customer_value - customers_in_top_k * offer_cost
    """
    y = np.asarray(y_true)[np.argsort(-np.asarray(y_score))]
    n = len(y)
    k = np.arange(1, n + 1)
    churners = np.cumsum(y)
    profit = churners * save_rate * customer_value - k * offer_cost
    return pd.DataFrame({"pct_targeted": k / n, "customers_targeted": k,
                         "churners_reached": churners, "expected_profit": profit})
