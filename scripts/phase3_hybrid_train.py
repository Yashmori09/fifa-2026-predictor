"""
Phase 3 v2 — Hybrid goal-scoring model.

Architecture:
  λ_home = XGB_h(features)   # expected goals for home, Poisson regression
  λ_away = XGB_a(features)   # expected goals for away
  → build joint scoreline PMF: P(i,j) = Poisson(λ_h,i) × Poisson(λ_a,j)
  → apply Dixon-Coles low-score correction (rho)
  → sum corners to W/D/L probabilities

This solves the structural draw-prediction problem: when λ_h ≈ λ_a,
draws emerge naturally from the Poisson distribution.

Compares against:
  • Phase 3 classifier (3-way XGB)
  • Pure Dixon-Coles
  • Baseline (pick higher squad)

Outputs:
  models/phase3v2_model.pkl  (the two regressors + rho)
  data/processed/phase3v2_results.csv
  docs/images/phase3/07_hybrid_calibration.png
  docs/images/phase3/08_hybrid_confusion.png
"""
import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
IMG = ROOT / "docs/images/phase3"

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    PHASE2_FEATURES, PHASE3_FEATURES,
    join_squad, join_static,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)

CLASS_NAMES = ["away_win", "draw", "home_win"]
MAX_GOALS = 10  # scoreline matrix dimension


# ── Hybrid model core ─────────────────────────────────────────────────

def fit_dc_rho(home_scores: np.ndarray, away_scores: np.ndarray,
               lam_h: np.ndarray, lam_a: np.ndarray) -> float:
    """Fit Dixon-Coles low-score correction parameter rho on training data."""
    def neg_log_likelihood(rho):
        rho = rho[0]
        # Per-match log-likelihood of observed (home, away) score under DC
        ll = 0.0
        for h, a, lh, la in zip(home_scores, away_scores, lam_h, lam_a):
            base = poisson.pmf(h, lh) * poisson.pmf(a, la)
            # DC correction only affects 0-0, 0-1, 1-0, 1-1
            if h == 0 and a == 0:
                base *= max(1 - lh * la * rho, 1e-10)
            elif h == 0 and a == 1:
                base *= max(1 + lh * rho, 1e-10)
            elif h == 1 and a == 0:
                base *= max(1 + la * rho, 1e-10)
            elif h == 1 and a == 1:
                base *= max(1 - rho, 1e-10)
            ll += np.log(max(base, 1e-15))
        return -ll
    # rho bounded for DC stability
    res = minimize(neg_log_likelihood, x0=[-0.05], method="L-BFGS-B",
                   bounds=[(-0.2, 0.2)])
    return float(res.x[0])


def lams_to_wdl(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Convert (λ_h, λ_a) + rho → P(away_win), P(draw), P(home_win)."""
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals, lam_h)
    pa = poisson.pmf(goals, lam_a)
    M = np.outer(ph, pa)
    # DC corrections on the 2×2 low-score block
    M[0, 0] *= max(1 - lam_h * lam_a * rho, 1e-10)
    M[0, 1] *= max(1 + lam_h * rho, 1e-10)
    M[1, 0] *= max(1 + lam_a * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()  # renormalize
    home_win = float(np.sum(M * (goals[:, None] > goals[None, :])))
    draw = float(np.sum(M * (goals[:, None] == goals[None, :])))
    away_win = max(0.0, 1.0 - home_win - draw)
    return np.array([away_win, draw, home_win])  # match label encoder order


def batch_lams_to_wdl(lam_h_arr: np.ndarray, lam_a_arr: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized W/D/L computation for a batch of (λ_h, λ_a)."""
    n = len(lam_h_arr)
    out = np.zeros((n, 3))
    for i in range(n):
        out[i] = lams_to_wdl(lam_h_arr[i], lam_a_arr[i], rho)
    return out


# ── Metrics ────────────────────────────────────────────────────────────

def all_metrics(name: str, y_true: np.ndarray, y_proba: np.ndarray, results: list):
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, np.argmax(y_proba, axis=1))
    n_cls = 3
    y_oh = np.eye(n_cls)[y_true]
    brier = float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))
    cum_p = np.cumsum(y_proba, axis=1)
    cum_t = np.cumsum(y_oh, axis=1)
    rps = float(np.mean(np.sum((cum_p - cum_t) ** 2, axis=1)) / (n_cls - 1))

    # Per-class recall
    cm = confusion_matrix(y_true, np.argmax(y_proba, axis=1), normalize="true")
    away_recall, draw_recall, home_recall = cm.diagonal()

    # ECE
    preds = np.argmax(y_proba, axis=1)
    confs = np.max(y_proba, axis=1)
    correct = (preds == y_true).astype(int)
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(y_true)) * abs(correct[m].mean() - confs[m].mean())

    results.append({"name": name, "log_loss": ll, "accuracy": acc, "brier": brier,
                    "rps": rps, "ece": ece, "away_recall": away_recall,
                    "draw_recall": draw_recall, "home_recall": home_recall})
    print(f"  {name:50s}  ll={ll:.4f}  acc={acc:.4f}  RPS={rps:.4f}  "
          f"draw_recall={draw_recall:.3f}  ECE={ece:.4f}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PHASE 3 v2 — Hybrid goal-scoring model (λ_h, λ_a → Poisson → W/D/L)")
    print("=" * 80)

    # Load data
    print("\n[Load]")
    train_dc = pd.read_csv(PROC / "train_dc.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")

    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()

    train = join_squad(train, tfy)
    train = join_static(train, sb, intl, chem)
    test = join_squad(test, tfy)
    test = join_static(test, sb, intl, chem)

    le = LabelEncoder()
    le.fit(["away_win", "draw", "home_win"])
    y_train_cls = le.transform(train["outcome"].values)
    y_test_cls = le.transform(test["outcome"].values)
    print(f"  train: {len(train):,}, test: {len(test):,}")
    print(f"  classes: {list(le.classes_)}")

    X_train = train[PHASE3_FEATURES].values
    X_test = test[PHASE3_FEATURES].values
    y_train_h = train["home_score"].astype(int).values
    y_train_a = train["away_score"].astype(int).values
    y_test_h = test["home_score"].astype(int).values
    y_test_a = test["away_score"].astype(int).values

    # ── Train Poisson regressors
    print("\n[Train] XGB Poisson regressors for home & away goals...")
    XGB_REG_P = dict(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        random_state=42, verbosity=0,
    )

    model_h = xgb.XGBRegressor(**XGB_REG_P)
    model_h.fit(X_train, y_train_h)
    lam_h_train = model_h.predict(X_train)
    lam_h_test = model_h.predict(X_test)
    print(f"  home goals model trained: train RMSE={np.sqrt(mean_squared_error(y_train_h, lam_h_train)):.3f}, "
          f"test RMSE={np.sqrt(mean_squared_error(y_test_h, lam_h_test)):.3f}")

    model_a = xgb.XGBRegressor(**XGB_REG_P)
    model_a.fit(X_train, y_train_a)
    lam_a_train = model_a.predict(X_train)
    lam_a_test = model_a.predict(X_test)
    print(f"  away goals model trained: train RMSE={np.sqrt(mean_squared_error(y_train_a, lam_a_train)):.3f}, "
          f"test RMSE={np.sqrt(mean_squared_error(y_test_a, lam_a_test)):.3f}")
    print(f"\n  λ_h test: mean={lam_h_test.mean():.2f}, range=[{lam_h_test.min():.2f}, {lam_h_test.max():.2f}]")
    print(f"  λ_a test: mean={lam_a_test.mean():.2f}, range=[{lam_a_test.min():.2f}, {lam_a_test.max():.2f}]")
    print(f"  actual h: mean={y_test_h.mean():.2f}, actual a: mean={y_test_a.mean():.2f}")

    # ── Fit DC rho on training residuals
    print("\n[Calibrate] Dixon-Coles rho on training data...")
    rho = fit_dc_rho(y_train_h, y_train_a, lam_h_train, lam_a_train)
    print(f"  fitted rho = {rho:.4f}  (negative → boosts draws, especially 1-1/0-0)")

    # ── Compute W/D/L probabilities for test set
    print("\n[Predict] converting (λ_h, λ_a) → W/D/L via Poisson + DC...")
    p_hybrid = batch_lams_to_wdl(lam_h_test, lam_a_test, rho)

    # Sanity: predicted-class distribution
    pred_cls = np.argmax(p_hybrid, axis=1)
    print(f"  predicted class distribution: "
          f"away_win={(pred_cls==0).mean():.3f}, "
          f"draw={(pred_cls==1).mean():.3f}, "
          f"home_win={(pred_cls==2).mean():.3f}")
    print(f"  actual class distribution:    "
          f"away_win={(y_test_cls==0).mean():.3f}, "
          f"draw={(y_test_cls==1).mean():.3f}, "
          f"home_win={(y_test_cls==2).mean():.3f}")

    # ── Compare to Phase 3 classifier
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)

    results = []

    # Load Phase 3 classifier predictions
    bundle_p3 = joblib.load(MODELS / "phase3_model.pkl")
    p_phase3 = bundle_p3["xgb"].predict_proba(test[bundle_p3["features_xgb"]].values)
    all_metrics("Phase 3 classifier (3-way XGB)", y_test_cls, p_phase3, results)

    # Hybrid
    all_metrics("Phase 3 v2 hybrid (λ→Poisson+DC)", y_test_cls, p_hybrid, results)

    # Pure Dixon-Coles (from existing DC ratings)
    dc_ratings = pd.read_csv(PROC / "dc_ratings.csv")
    with open(MODELS / "dc_params.json") as f:
        dp = json.load(f)
    rmap = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}

    def dc_pred(row):
        h, a = row["home_team"], row["away_team"]
        if h not in rmap or a not in rmap:
            return np.array([0.45, 0.25, 0.30])
        h_att, h_def = rmap[h]
        a_att, a_def = rmap[a]
        ha = dp["home_adv"] if not bool(row.get("neutral.1", row.get("neutral", True))) else 0.0
        lh = np.exp(h_att + a_def + ha)
        la = np.exp(a_att + h_def)
        return lams_to_wdl(lh, la, dp["rho"])

    p_pure_dc = np.array([dc_pred(r) for _, r in test.iterrows()])
    all_metrics("Pure Dixon-Coles", y_test_cls, p_pure_dc, results)

    # Blend: 4× hybrid + 1× phase3 (might be best of both)
    p_blend = (4 * p_hybrid + 1 * p_phase3) / 5
    all_metrics("Hybrid×4 + Phase3×1 blend", y_test_cls, p_blend, results)

    # ── Save
    rdf = pd.DataFrame(results)
    rdf.to_csv(PROC / "phase3v2_results.csv", index=False)

    joblib.dump({
        "model_home": model_h,
        "model_away": model_a,
        "rho": rho,
        "features": PHASE3_FEATURES,
        "label_encoder": le,
    }, MODELS / "phase3v2_model.pkl")
    print(f"\n  Saved: models/phase3v2_model.pkl + data/processed/phase3v2_results.csv")

    # ── Plots
    print("\n[Plots]")

    # Calibration comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0, 1, 11)
    for cls_idx, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        for label, p, marker in [("Phase 3", p_phase3, "o"), ("Hybrid", p_hybrid, "s")]:
            probs = p[:, cls_idx]
            actuals = (y_test_cls == cls_idx).astype(int)
            bc, br = [], []
            for i in range(10):
                m = (probs >= bins[i]) & (probs < bins[i + 1])
                if m.sum() >= 5:
                    bc.append((bins[i] + bins[i + 1]) / 2)
                    br.append(actuals[m].mean())
            ax.plot(bc, br, marker + "-", label=label, linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"Class: {name}")
        ax.set_xlabel("Predicted prob")
        ax.set_ylabel("Actual freq")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Calibration: Phase 3 classifier vs Hybrid (goal-scoring) model")
    plt.tight_layout()
    plt.savefig(IMG / "07_hybrid_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 07_hybrid_calibration.png")

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p, name in [(axes[0], p_phase3, "Phase 3 classifier"),
                         (axes[1], p_hybrid, "Hybrid model")]:
        cm = confusion_matrix(y_test_cls, np.argmax(p, axis=1), normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    cbar_kws={"label": "frequency"})
        ax.set_title(f"{name} — confusion (row-normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(IMG / "08_hybrid_confusion.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 08_hybrid_confusion.png")

    # Predicted draw probability distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(p_phase3[:, 1], bins=30, color="#F39C12", alpha=0.8, edgecolor="black")
    axes[0].axvline(0.5, color="red", linestyle="--", label="argmax threshold")
    axes[0].set_title("Phase 3 classifier — P(draw) distribution\n(capped at 0.5, never picks draw)")
    axes[0].set_xlabel("P(draw)"); axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1); axes[0].legend()
    axes[1].hist(p_hybrid[:, 1], bins=30, color="#27AE60", alpha=0.8, edgecolor="black")
    axes[1].axvline(0.5, color="red", linestyle="--", label="argmax threshold")
    axes[1].set_title("Hybrid model — P(draw) distribution\n(can predict high-prob draws)")
    axes[1].set_xlabel("P(draw)"); axes[1].set_ylabel("Count")
    axes[1].set_xlim(0, 1); axes[1].legend()
    plt.tight_layout()
    plt.savefig(IMG / "09_hybrid_draw_dist.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 09_hybrid_draw_dist.png")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(rdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
