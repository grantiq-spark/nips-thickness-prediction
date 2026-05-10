"""
PDP/ICE, simplified model, leakage checks, uncertainty, and publication figures
for DWT-D-26-00307 / DES-D-26-01032 revised manuscript.

Expected input: CSV with one row per production record.
Required columns are configured in CONFIG below.
Run:
    python scripts/pdp_ice_analysis.py --input data/raw/anonymized_nips_thickness.csv --out results

Notes:
- Uses GroupKFold/TimeSeriesSplit alternatives in addition to shuffled CV.
- Fits preprocessing inside CV pipelines to avoid leakage.
- Produces white-background, 300 dpi figures.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


@dataclass
class Config:
    target: str = "dry_thickness_um"
    date_col: str = "production_date"
    product_col: str = "product_family"
    lot_col: str = "lot_id"
    numeric_cols: Tuple[str, ...] = (
        "viscosity_cp", "casting_speed_m_min", "doctor_blade_gap_um",
        "dope_flow_rate_ml_min", "solution_concentration_pct", "solid_content_pct",
        "nips_bath1_temp_c", "nips_bath2_temp_c", "dry1_temp_c", "dry2_temp_c", "dry3_temp_c",
    )
    random_state: int = 42

CONFIG = Config()


def set_pub_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9
    v = out["viscosity_cp"]
    s = out["casting_speed_m_min"]
    g = out["doctor_blade_gap_um"]
    q = out["dope_flow_rate_ml_min"].replace(0, np.nan)
    out["viscosity_x_speed"] = v * s
    out["viscosity_x_gap"] = v * g
    out["viscosity_div_flow"] = v / (q + eps)
    out["speed_x_gap"] = s * g
    out["log_viscosity"] = np.log1p(v.clip(lower=0))
    out["viscosity_sq"] = v ** 2
    return out


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [CONFIG.target, CONFIG.product_col, *CONFIG.numeric_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if CONFIG.date_col in df.columns:
        df[CONFIG.date_col] = pd.to_datetime(df[CONFIG.date_col], errors="coerce")
    return add_engineered_features(df)


def make_preprocessor(df: pd.DataFrame, imputer="median") -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric = list(CONFIG.numeric_cols) + [
        "viscosity_x_speed", "viscosity_x_gap", "viscosity_div_flow",
        "speed_x_gap", "log_viscosity", "viscosity_sq",
    ]
    categorical = [CONFIG.product_col]
    if imputer == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputer == "mean":
        num_imputer = SimpleImputer(strategy="mean")
    else:
        num_imputer = SimpleImputer(strategy="median")
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", num_imputer)]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return pre, numeric, categorical


def model_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.035, max_depth=5,
        min_samples_leaf=3, subsample=0.85, random_state=CONFIG.random_state,
    )


def make_pipeline(model=None, imputer="median") -> Pipeline:
    pre, _, _ = make_preprocessor(pd.DataFrame(), imputer=imputer)
    return Pipeline([("pre", pre), ("model", model if model is not None else model_gbr())])


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_um": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_um": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def missingness_report(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    cols = list(CONFIG.numeric_cols) + [CONFIG.target, CONFIG.product_col]
    rep = []
    for c in cols:
        rep.append({"feature": c, "missing_n": int(df[c].isna().sum()), "missing_pct": float(df[c].isna().mean() * 100)})
    r = pd.DataFrame(rep).sort_values("missing_pct", ascending=False)
    r.to_csv(outdir / "missingness_statistics.csv", index=False)
    return r


def evaluate_cv(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float)
    rows = []

    # Random shuffled CV - retained for comparability with original submission.
    cv_random = KFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)
    pred = cross_val_predict(make_pipeline(model_gbr()), X, y, cv=cv_random)
    rows.append({"validation": "stratified/random 5-fold surrogate", **metrics(y, pred)})

    # Lot-aware CV where lot_id exists.
    if CONFIG.lot_col in df.columns and df[CONFIG.lot_col].nunique() >= 5:
        cv_group = GroupKFold(n_splits=5)
        pred = cross_val_predict(make_pipeline(model_gbr()), X, y, cv=cv_group, groups=df[CONFIG.lot_col])
        rows.append({"validation": "lot-grouped 5-fold", **metrics(y, pred)})

    # Time-aware validation.
    if CONFIG.date_col in df.columns and df[CONFIG.date_col].notna().sum() > 100:
        sdf = df.sort_values(CONFIG.date_col).reset_index(drop=True)
        Xs = sdf.drop(columns=[CONFIG.target])
        ys = sdf[CONFIG.target].astype(float)
        tscv = TimeSeriesSplit(n_splits=5)
        preds = np.full(len(sdf), np.nan)
        for tr, te in tscv.split(Xs):
            pipe = make_pipeline(model_gbr())
            pipe.fit(Xs.iloc[tr], ys.iloc[tr])
            preds[te] = pipe.predict(Xs.iloc[te])
        mask = ~np.isnan(preds)
        rows.append({"validation": "forward-chaining time split", **metrics(ys[mask], preds[mask])})

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "validation_strategy_comparison.csv", index=False)
    return res


def imputation_sensitivity(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)
    rows = []
    for method in ["median", "mean", "knn"]:
        pred = cross_val_predict(make_pipeline(model_gbr(), imputer=method), X, y, cv=cv)
        rows.append({"imputation": method, **metrics(y, pred)})
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "imputation_sensitivity.csv", index=False)
    return res


def compare_tree_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    models = {"GBR-500": model_gbr(), "RF-500": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)}
    if XGBRegressor is not None:
        models["XGBoost-500"] = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.85,
                                             colsample_bytree=0.85, random_state=42, objective="reg:squarederror")
    if LGBMRegressor is not None:
        models["LightGBM-500"] = LGBMRegressor(n_estimators=500, learning_rate=0.035, random_state=42)
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, m in models.items():
        pred = cross_val_predict(make_pipeline(m), X, y, cv=cv)
        rows.append({"model": name, **metrics(y, pred)})
    res = pd.DataFrame(rows).sort_values("R2", ascending=False)
    res.to_csv(outdir / "tree_model_comparison.csv", index=False)
    return res


def fit_final(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float)
    pipe = make_pipeline(model_gbr())
    pipe.fit(X, y)
    return pipe


def make_pdp_ice(df: pd.DataFrame, outdir: Path) -> None:
    X = df.drop(columns=[CONFIG.target])
    pipe = fit_final(df)
    features = ["viscosity_x_speed", "solution_concentration_pct", ("viscosity_x_speed", "solution_concentration_pct")]
    for feat in features:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        PartialDependenceDisplay.from_estimator(
            pipe, X, [feat], kind="both" if isinstance(feat, str) else "average",
            subsample=80, random_state=42, ax=ax, grid_resolution=30,
        )
        title = "PDP/ICE: " + (feat if isinstance(feat, str) else f"{feat[0]} x {feat[1]}")
        ax.set_title(title)
        fig.tight_layout()
        safe = title.replace("/", "_").replace(": ", "_").replace(" ", "_").replace("x", "x")
        fig.savefig(outdir / f"{safe}.png")
        plt.close(fig)


def residual_analysis(df: pd.DataFrame, outdir: Path) -> None:
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(make_pipeline(model_gbr()), X, y, cv=cv)
    r = y - pred
    resdf = df[[CONFIG.product_col, CONFIG.target]].copy()
    for col in ["viscosity_x_speed", "solution_concentration_pct", "casting_speed_m_min", "viscosity_cp", "doctor_blade_gap_um"]:
        if col in df.columns:
            resdf[col] = df[col]
    resdf["prediction_um"] = pred
    resdf["residual_um"] = r
    resdf["abs_residual_um"] = np.abs(r)
    q90 = resdf["abs_residual_um"].quantile(0.90)
    resdf["high_residual_flag"] = resdf["abs_residual_um"] >= q90
    resdf.to_csv(outdir / "cross_validated_residuals.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(pred, r, s=18, alpha=0.7)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Predicted thickness (um)")
    ax.set_ylabel("Residual (actual - predicted, um)")
    ax.set_title("Residual map with 90th percentile error boundary")
    fig.tight_layout()
    fig.savefig(outdir / "Fig_residual_map_white.png")
    plt.close(fig)


def bootstrap_uncertainty(df: pd.DataFrame, outdir: Path, n_boot=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = df.drop(columns=[CONFIG.target])
    y = df[CONFIG.target].astype(float).reset_index(drop=True)
    preds = []
    for _ in range(n_boot):
        idx = rng.choice(len(df), len(df), replace=True)
        pipe = make_pipeline(model_gbr())
        pipe.fit(X.iloc[idx], y.iloc[idx])
        preds.append(pipe.predict(X))
    arr = np.vstack(preds)
    out = pd.DataFrame({
        "actual_um": y,
        "pred_mean_um": arr.mean(axis=0),
        "pred_p05_um": np.percentile(arr, 5, axis=0),
        "pred_p95_um": np.percentile(arr, 95, axis=0),
        "pred_sd_um": arr.std(axis=0),
    })
    out.to_csv(outdir / "bootstrap_prediction_intervals.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    order = np.argsort(out["actual_um"].values)
    x = np.arange(len(out))
    ax.plot(x, out["actual_um"].values[order], label="Actual", linewidth=1)
    ax.plot(x, out["pred_mean_um"].values[order], label="Predicted mean", linewidth=1)
    ax.fill_between(x, out["pred_p05_um"].values[order], out["pred_p95_um"].values[order], alpha=0.2, label="90% interval")
    ax.set_xlabel("Samples sorted by actual thickness")
    ax.set_ylabel("Thickness (um)")
    ax.set_title("Bootstrap uncertainty band")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "Fig_bootstrap_uncertainty_white.png")
    plt.close(fig)
    return out


def simplified_model(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    # Deployment-oriented model using top physically interpretable variables.
    keep = [CONFIG.target, CONFIG.product_col, "viscosity_x_speed", "viscosity_div_flow", "solution_concentration_pct", "dope_flow_rate_ml_min", "log_viscosity"]
    sdf = df[[c for c in keep if c in df.columns]].copy()
    X = sdf.drop(columns=[CONFIG.target])
    y = sdf[CONFIG.target].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(make_pipeline(model_gbr()), X, y, cv=cv)
    res = pd.DataFrame([{"model": "Simplified top-feature GBR", "n_features": X.shape[1], **metrics(y, pred)}])
    res.to_csv(outdir / "simplified_model_performance.csv", index=False)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    set_pub_style()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.input))
    missingness_report(df, outdir)
    evaluate_cv(df, outdir)
    imputation_sensitivity(df, outdir)
    compare_tree_models(df, outdir)
    make_pdp_ice(df, outdir)
    residual_analysis(df, outdir)
    bootstrap_uncertainty(df, outdir, n_boot=args.bootstrap)
    simplified_model(df, outdir)
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"n_records": int(len(df)), "columns": list(df.columns)}, f, indent=2)


if __name__ == "__main__":
    main()
