"""
One-class Gaussian Process pipeline for fetal well-being prediction.

What this script does:
- loads feature table + labels
- expands labels (each label → 3 segments)
- keeps only CAT-1 and CAT-3
- splits CAT-1 into train/test (no leakage)
- makes CAT-1 test size match CAT-3
- trains GP only on CAT-1
- evaluates on CAT-1 test + CAT-3
- saves predictions, scores, and metrics

Assumptions:
- feature extraction already done
- each label corresponds to 3 segments
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Feature list
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "baseline_fhr",
    "num_accelerations",
    "num_decelerations",
    "accel_duration_seconds",
    "decel_duration_seconds",
    "mean_fhr",
    "median_fhr",
    "std_fhr",
    "min_fhr",
    "max_fhr",
    "range_fhr",
    "rmssd",
    "peak_frequency",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "approx_entropy",
    "sample_entropy",
    "dfa",
    "variance_fhr",
    "iqr_fhr",
    "percentile_25",
    "percentile_75",
]


# ------------------------------------------------------------
# Simple containers
# ------------------------------------------------------------

@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: np.ndarray
    metadata_test: pd.DataFrame


@dataclass
class EvalResults:
    predictions: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    anomaly_score: np.ndarray
    confidence: np.ndarray
    threshold: float
    var_min: float
    var_max: float
    metrics: dict[str, float]


# ------------------------------------------------------------
# GP model
# ------------------------------------------------------------

class OneClassGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant.data.fill_(0.0)

        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        # light priors (kept simple)
        self.covar_module.base_kernel.register_prior(
            "lengthscale_prior",
            gpytorch.priors.GammaPrior(2.0, 0.5),
            "lengthscale",
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def load_features(path: Path) -> pd.DataFrame:
    df = load_table(path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.copy()


def load_labels(path: Path) -> pd.DataFrame:
    df = load_table(path)

    if "cat" not in df.columns:
        raise ValueError("Label file must contain 'cat' column")

    return df.copy()


# ------------------------------------------------------------
# Label handling
# ------------------------------------------------------------

def expand_labels(labels_df, segments_per_label=3):
    return pd.Series(
        np.repeat(labels_df["cat"].values, segments_per_label),
        name="cat",
    )


def attach_labels(features_df, labels_path=None, segments_per_label=3):
    df = features_df.copy()

    if "cat" in df.columns:
        return df

    if labels_path is None:
        raise ValueError("No labels found")

    labels_df = load_labels(labels_path)
    expanded = expand_labels(labels_df, segments_per_label)

    if len(expanded) != len(df):
        raise ValueError("Label/feature length mismatch")

    df["cat"] = expanded.values
    return df


def filter_data(df):
    df = df.dropna(subset=["cat"])
    return df[df["cat"].isin(["CAT-1", "CAT-3"])].copy()


# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

def split_data(df, random_state=42):
    normal = df[df["cat"] == "CAT-1"].reset_index(drop=True)
    abnormal = df[df["cat"] == "CAT-3"]

    n_abnormal = len(abnormal)

    if n_abnormal == 0:
        raise ValueError("No CAT-3 samples found")

    if len(normal) <= n_abnormal:
        raise ValueError("Not enough CAT-1 samples")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(normal), size=n_abnormal, replace=False)

    train_mask = np.ones(len(normal), dtype=bool)
    train_mask[idx] = False

    x_train = normal.loc[train_mask, FEATURE_COLUMNS]
    x_test_normal = normal.loc[idx, FEATURE_COLUMNS]

    x_test = pd.concat(
        [x_test_normal, abnormal[FEATURE_COLUMNS]],
        axis=0,
    ).reset_index(drop=True)

    metadata_test = pd.concat(
        [normal.loc[idx], abnormal],
        axis=0,
    ).reset_index(drop=True)

    y_test = np.array([1] * len(x_test_normal) + [0] * len(abnormal))

    return SplitData(x_train, x_test, y_test, metadata_test)


# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

def scale_features(x_train, x_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(x_train),
        scaler.transform(x_test),
        scaler,
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_gp(x_train, iters=300):
    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.ones(len(train_x))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = OneClassGP(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iters):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(f"Iter {i} | Loss {loss.item():.4f}")

    return model, likelihood


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict(model, likelihood, x):
    model.eval()
    likelihood.eval()

    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))

    return pred.mean.numpy(), pred.variance.numpy()


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(model, likelihood, x_train, x_test, y_test):
    _, train_var = predict(model, likelihood, x_train)

    threshold = np.percentile(train_var, 95)
    vmin, vmax = train_var.min(), train_var.max()

    mean, var = predict(model, likelihood, x_test)

    preds = (var < threshold).astype(int)
    confidence = 1 - (var - vmin) / (vmax - vmin + 1e-8)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_test, -var)
    except:
        metrics["auroc"] = np.nan

    return EvalResults(
        preds,
        mean,
        var,
        var,
        confidence,
        threshold,
        vmin,
        vmax,
        metrics,
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

def save_results(out_dir, split, res):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = split.metadata_test.copy()
    df["y_true"] = split.y_test
    df["prediction"] = res.predictions
    df["variance"] = res.var
    df["confidence"] = res.confidence

    df.to_excel(out_dir / "results.xlsx", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res.metrics, f, indent=2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))

    args = parser.parse_args()

    print("Loading data...")
    df = load_features(args.features)

    df = attach_labels(df, args.labels)
    df = filter_data(df)

    split = split_data(df)
    x_train, x_test, _ = scale_features(split.x_train, split.x_test)

    print("Training GP...")
    model, likelihood = train_gp(x_train)

    print("Evaluating...")
    res = evaluate(model, likelihood, x_train, x_test, split.y_test)

    print("Metrics:")
    for k, v in res.metrics.items():
        print(f"{k}: {v:.4f}")

    save_results(args.output, split, res)


if __name__ == "__main__":
    main()
