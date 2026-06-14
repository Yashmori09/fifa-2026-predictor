"""
Phase D-1 — Scoreline prediction evaluation.

The hybrid model predicts (λ_h, λ_a) which → full scoreline matrix
P(home_score=i, away_score=j) for (i,j) up to MAX_GOALS. The most-likely
scoreline is argmax of this matrix.

We evaluate on the 748-match holdout:
  • Exact scoreline accuracy:  predicted == actual
  • Within-1 scoreline accuracy: |pred_h - act_h| ≤ 1 AND |pred_a - act_a| ≤ 1
  • Most-common predicted scorelines (does the model collapse to 1-0?)
  • Per-scoreline calibration: when model predicts 2-1, how often does 2-1 happen?

Baselines:
  • Always predict 1-0 (most common scoreline)
  • Always predict 1-1
  • Random sample from training scoreline distribution

Also breaks down by:
  • Match type (qualifier, tournament)
  • |ELO diff| bucket
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    PHASE3_FEATURES, join_squad, join_static,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)

MAX_GOALS = 10


def scoreline_matrix(lh, la, rho):
    """Return (MAX_GOALS+1, MAX_GOALS+1) joint P(home_score, away_score)."""
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= max(1 - lh * la * rho, 1e-10)
    M[0, 1] *= max(1 + lh * rho, 1e-10)
    M[1, 0] *= max(1 + la * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    return M


def most_likely_score(M):
    """Argmax of scoreline matrix."""
    idx = np.unravel_index(np.argmax(M), M.shape)
    return int(idx[0]), int(idx[1])


def fit_rho(hs, as_, lhs, las):
    def nll(r):
        r = r[0]
        ll = 0
        for h, a, lh, la in zip(hs, as_, lhs, las):
            b = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if h == 0 and a == 0: b *= max(1 - lh * la * r, 1e-10)
            elif h == 0 and a == 1: b *= max(1 + lh * r, 1e-10)
            elif h == 1 and a == 0: b *= max(1 + la * r, 1e-10)
            elif h == 1 and a == 1: b *= max(1 - r, 1e-10)
            ll += np.log(max(b, 1e-15))
        return -ll
    res = minimize(nll, x0=[-0.05], method="L-BFGS-B", bounds=[(-0.2, 0.2)])
    return float(res.x[0])


def main():
    print("=" * 90)
    print("PHASE D-1 — Scoreline prediction evaluation")
    print("=" * 90)

    # Load + join features (same recipe as Phase 3 hybrid)
    train_dc = pd.read_csv(PROC / "train_dc_v2.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc_v2.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    pool = join_squad(pool, tfy)
    pool = join_static(pool, sb, intl, chem)

    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()
    print(f"\n  train: {len(train):,}, test: {len(test):,}")

    X_train = train[PHASE3_FEATURES].values
    X_test = test[PHASE3_FEATURES].values
    y_tr_h = train["home_score"].astype(int).values
    y_tr_a = train["away_score"].astype(int).values

    # Train hybrid
    print("\n[Train] hybrid Poisson regressors...")
    XGB_REG_P = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                     objective="count:poisson", eval_metric="poisson-nloglik",
                     random_state=42, verbosity=0)
    mh = xgb.XGBRegressor(**XGB_REG_P); mh.fit(X_train, y_tr_h)
    ma = xgb.XGBRegressor(**XGB_REG_P); ma.fit(X_train, y_tr_a)
    lh_tr = mh.predict(X_train); lh_te = mh.predict(X_test)
    la_tr = ma.predict(X_train); la_te = ma.predict(X_test)
    rho = fit_rho(y_tr_h, y_tr_a, lh_tr, la_tr)
    print(f"  rho = {rho:.4f}")
    print(f"  λ_h mean: train={lh_tr.mean():.2f}, test={lh_te.mean():.2f}, actual_test={test['home_score'].mean():.2f}")
    print(f"  λ_a mean: train={la_tr.mean():.2f}, test={la_te.mean():.2f}, actual_test={test['away_score'].mean():.2f}")

    # ── Score every holdout match
    print("\n[Predict] computing scoreline matrices for 748 holdout matches...")
    actual_scores = list(zip(test["home_score"].astype(int).values,
                             test["away_score"].astype(int).values))
    predicted_scores = []
    score_log_probs = []  # log P(actual scoreline) under model
    for lh, la, (ah, aa) in zip(lh_te, la_te, actual_scores):
        M = scoreline_matrix(lh, la, rho)
        # cap actual at MAX_GOALS for indexing
        ah_c = min(ah, MAX_GOALS)
        aa_c = min(aa, MAX_GOALS)
        score_log_probs.append(np.log(max(M[ah_c, aa_c], 1e-15)))
        predicted_scores.append(most_likely_score(M))

    score_log_probs = np.array(score_log_probs)
    print(f"  mean scoreline log loss: {-score_log_probs.mean():.4f}")

    # ── Exact and within-1 accuracy
    print("\n" + "=" * 90)
    print("SCORELINE ACCURACY — Phase 3 hybrid")
    print("=" * 90)
    exact = sum(1 for (ph, pa), (ah, aa) in zip(predicted_scores, actual_scores) if ph == ah and pa == aa)
    within1 = sum(1 for (ph, pa), (ah, aa) in zip(predicted_scores, actual_scores)
                  if abs(ph - ah) <= 1 and abs(pa - aa) <= 1)
    n = len(actual_scores)
    print(f"  Exact:    {exact}/{n} = {exact/n*100:5.1f}%")
    print(f"  Within 1: {within1}/{n} = {within1/n*100:5.1f}%")

    # ── Baselines
    print("\n  Baselines:")
    for baseline_score, label in [((1, 0), "always 1-0"),
                                   ((1, 1), "always 1-1"),
                                   ((2, 1), "always 2-1"),
                                   ((0, 0), "always 0-0")]:
        e = sum(1 for s in actual_scores if s == baseline_score)
        w = sum(1 for s in actual_scores if abs(s[0]-baseline_score[0])<=1 and abs(s[1]-baseline_score[1])<=1)
        print(f"  {label:14s}: exact={e:>3}/{n} = {e/n*100:5.1f}%  within1={w:>3}/{n} = {w/n*100:5.1f}%")

    # Random sample from training scoreline distribution
    train_scores = list(zip(y_tr_h, y_tr_a))
    score_counter = Counter(train_scores)
    most_common_train = score_counter.most_common(10)
    print(f"\n  Top 10 training scorelines:")
    for s, c in most_common_train:
        print(f"    {s[0]}-{s[1]}: {c:>4} ({c/len(train_scores)*100:4.1f}%)")

    # ── What does the model predict as most-common?
    print(f"\n  Top 10 model-predicted most-likely scorelines (out of {n} holdout matches):")
    pred_counter = Counter(predicted_scores)
    for s, c in pred_counter.most_common(10):
        actual_freq = sum(1 for a in actual_scores if a == s)
        print(f"    {s[0]}-{s[1]}: predicted {c:>3} times, actual freq = {actual_freq}")

    # ── By tournament category
    print("\n  Breakdown by match type:")
    test = test.reset_index(drop=True)
    test["pred_score"] = predicted_scores
    test["actual_score"] = actual_scores
    test["exact"] = [(ph, pa) == (ah, aa) for (ph, pa), (ah, aa) in zip(predicted_scores, actual_scores)]
    test["within1"] = [abs(ph - ah) <= 1 and abs(pa - aa) <= 1
                       for (ph, pa), (ah, aa) in zip(predicted_scores, actual_scores)]
    by_cat = test.groupby("tournament_category").agg(
        n=("exact", "size"), exact_acc=("exact", "mean"), within1_acc=("within1", "mean")
    ).round(3)
    print(by_cat.to_string())

    # ── By ELO diff bucket
    print("\n  Breakdown by |ELO diff|:")
    test["abs_elo"] = test["elo_diff"].abs()
    test["bucket"] = pd.cut(test["abs_elo"], bins=[0, 50, 100, 200, 400, 9999],
                            labels=["close (0-50)", "moderate (50-100)",
                                    "clear (100-200)", "lopsided (200-400)", "blowout (400+)"])
    by_b = test.groupby("bucket", observed=True).agg(
        n=("exact", "size"), exact_acc=("exact", "mean"), within1_acc=("within1", "mean")
    ).round(3)
    print(by_b.to_string())

    # ── Top 10 most-wrong scoreline predictions
    print("\n  Top 10 worst scoreline misses (lowest P(actual scoreline)):")
    test["score_log_prob"] = score_log_probs
    worst = test.nsmallest(10, "score_log_prob")[
        ["date", "home_team", "away_team", "actual_score", "pred_score", "score_log_prob"]
    ]
    print(worst.to_string(index=False))

    # Save model bundle for the simulation step
    import joblib
    joblib.dump({
        "model_home": mh, "model_away": ma, "rho": rho,
        "features": PHASE3_FEATURES,
    }, ROOT / "models/phase3_hybrid_clean.pkl")
    print(f"\n  Saved hybrid model: models/phase3_hybrid_clean.pkl")


if __name__ == "__main__":
    main()
