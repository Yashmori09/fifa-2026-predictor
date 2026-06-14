"""
Phase 3 — re-evaluate ALL models with the leakage-free DC features
(train_dc_v2.csv + test_dc_v2.csv + dc_ratings_v2.csv).

Models compared on the 2025-06-11 → 2026-03-31 holdout:
  1. Phase 3 classifier (3-way XGB)        — using v2 features
  2. Phase 3 hybrid (XGB Poisson → W/D/L)  — using v2 features
  3. Pure Dixon-Coles                       — using v2 ratings (no longer leaked)
  4. Blends

Key metrics:
  log loss, accuracy, RPS, ECE, per-class recall (esp. draw_recall)
"""
import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    PHASE2_FEATURES, PHASE3_FEATURES,
    join_squad, join_static,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)

CLASS_NAMES = ["away_win", "draw", "home_win"]
MAX_GOALS = 10


def lams_to_wdl(lam_h, lam_a, rho):
    goals = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    M[0, 0] *= max(1 - lam_h * lam_a * rho, 1e-10)
    M[0, 1] *= max(1 + lam_h * rho, 1e-10)
    M[1, 0] *= max(1 + lam_a * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    hw = float(np.sum(M * (goals[:, None] > goals[None, :])))
    dr = float(np.sum(M * (goals[:, None] == goals[None, :])))
    aw = max(0.0, 1.0 - hw - dr)
    return np.array([aw, dr, hw])


def batch_lams_to_wdl(lam_h, lam_a, rho):
    out = np.zeros((len(lam_h), 3))
    for i in range(len(lam_h)):
        out[i] = lams_to_wdl(lam_h[i], lam_a[i], rho)
    return out


def fit_dc_rho(home_scores, away_scores, lam_h, lam_a):
    def nll(rho_arr):
        rho = rho_arr[0]
        ll = 0.0
        for h, a, lh, la in zip(home_scores, away_scores, lam_h, lam_a):
            base = poisson.pmf(h, lh) * poisson.pmf(a, la)
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
    res = minimize(nll, x0=[-0.05], method="L-BFGS-B", bounds=[(-0.2, 0.2)])
    return float(res.x[0])


def metrics(name, y_true, y_proba, results):
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, np.argmax(y_proba, axis=1))
    y_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))
    cum_p = np.cumsum(y_proba, axis=1)
    cum_t = np.cumsum(y_oh, axis=1)
    rps = float(np.mean(np.sum((cum_p - cum_t) ** 2, axis=1)) / 2)
    cm = confusion_matrix(y_true, np.argmax(y_proba, axis=1), normalize="true")
    aw_r, dr_r, hw_r = cm.diagonal()
    preds = np.argmax(y_proba, axis=1)
    confs = np.max(y_proba, axis=1)
    correct = (preds == y_true).astype(int)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(y_true)) * abs(correct[m].mean() - confs[m].mean())
    results.append({"name": name, "ll": ll, "acc": acc, "brier": brier, "rps": rps,
                    "ece": ece, "aw_r": aw_r, "dr_r": dr_r, "hw_r": hw_r})
    print(f"  {name:50s}  ll={ll:.4f}  acc={acc:.4f}  RPS={rps:.4f}  "
          f"draw_r={dr_r:.3f}  ECE={ece:.4f}")


def main():
    print("=" * 80)
    print("PHASE 3 — clean re-evaluation with leakage-free DC")
    print("=" * 80)

    # ── Load clean data
    train_dc = pd.read_csv(PROC / "train_dc_v2.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc_v2.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()
    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()
    print(f"\n  train: {len(train):,}, test: {len(test):,}")

    # ── Join all features
    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    train = join_squad(train, tfy); train = join_static(train, sb, intl, chem)
    test = join_squad(test, tfy); test = join_static(test, sb, intl, chem)

    le = LabelEncoder()
    le.fit(["away_win", "draw", "home_win"])
    y_train = le.transform(train["outcome"].values)
    y_test = le.transform(test["outcome"].values)

    X_train_p2 = train[PHASE2_FEATURES].values
    X_test_p2 = test[PHASE2_FEATURES].values
    X_train_p3 = train[PHASE3_FEATURES].values
    X_test_p3 = test[PHASE3_FEATURES].values

    XGB_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    XGB_REG_P = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                     objective="count:poisson", eval_metric="poisson-nloglik",
                     random_state=42, verbosity=0)

    results = []

    # ── 1. Phase 2 classifier (BASE+SQUAD only, no new B5)
    print("\n[1] Phase 2 classifier (BASE + SQUAD features, clean DC)")
    xgb_p2 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_p2.fit(X_train_p2, y_train)
    p_p2 = xgb_p2.predict_proba(X_test_p2)
    metrics("Phase 2 (3-way XGB, no B5)", y_test, p_p2, results)

    # ── 2. Phase 3 classifier (BASE+SQUAD+B5)
    print("\n[2] Phase 3 classifier (+ B5 features, clean DC)")
    xgb_p3 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_p3.fit(X_train_p3, y_train)
    p_p3 = xgb_p3.predict_proba(X_test_p3)
    metrics("Phase 3 (3-way XGB + B5)", y_test, p_p3, results)

    # ── 3. Hybrid (Poisson regressors + DC correction)
    print("\n[3] Hybrid (XGB Poisson regressors → λ_h, λ_a → W/D/L)")
    y_train_h = train["home_score"].astype(int).values
    y_train_a = train["away_score"].astype(int).values

    model_h = xgb.XGBRegressor(**XGB_REG_P)
    model_h.fit(X_train_p3, y_train_h)
    lam_h_train = model_h.predict(X_train_p3)
    lam_h_test = model_h.predict(X_test_p3)

    model_a = xgb.XGBRegressor(**XGB_REG_P)
    model_a.fit(X_train_p3, y_train_a)
    lam_a_train = model_a.predict(X_train_p3)
    lam_a_test = model_a.predict(X_test_p3)

    print(f"  λ_h test mean={lam_h_test.mean():.2f}, λ_a test mean={lam_a_test.mean():.2f}")
    print(f"  actual h mean={test['home_score'].mean():.2f}, actual a mean={test['away_score'].mean():.2f}")
    print(f"  RMSE h={np.sqrt(mean_squared_error(test['home_score'], lam_h_test)):.3f}, "
          f"a={np.sqrt(mean_squared_error(test['away_score'], lam_a_test)):.3f}")

    rho_hybrid = fit_dc_rho(y_train_h, y_train_a, lam_h_train, lam_a_train)
    print(f"  fitted rho = {rho_hybrid:.4f}")

    p_hybrid = batch_lams_to_wdl(lam_h_test, lam_a_test, rho_hybrid)
    metrics("Hybrid (XGB-Poisson + DC)", y_test, p_hybrid, results)

    # ── 4. Pure Dixon-Coles (now with clean v2 ratings)
    print("\n[4] Pure Dixon-Coles (using leakage-free v2 ratings)")
    dc_ratings = pd.read_csv(PROC / "dc_ratings_v2.csv")
    with open(MODELS / "dc_params_v2.json") as f:
        dp = json.load(f)
    rmap = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}

    def dc_pred(row):
        h, a = row["home_team"], row["away_team"]
        if h not in rmap or a not in rmap:
            return np.array([0.45, 0.25, 0.30])
        h_att, h_def = rmap[h]
        a_att, a_def = rmap[a]
        neutral = bool(row.get("neutral.1", row.get("neutral", True)))
        ha = dp["home_adv"] if not neutral else 0.0
        lh = float(np.exp(h_att + a_def + ha))
        la = float(np.exp(a_att + h_def))
        return lams_to_wdl(lh, la, dp["rho"])

    p_dc = np.array([dc_pred(r) for _, r in test.iterrows()])
    metrics("Pure Dixon-Coles (clean)", y_test, p_dc, results)

    # ── 5. Blends
    print("\n[5] Blends")
    p_p3_dc = (5 * p_p3 + 1 * p_dc) / 6
    metrics("Phase 3 × 5 + DC × 1", y_test, p_p3_dc, results)

    p_hyb_dc = (5 * p_hybrid + 1 * p_dc) / 6
    metrics("Hybrid × 5 + DC × 1", y_test, p_hyb_dc, results)

    p_hyb_p3 = (3 * p_hybrid + 1 * p_p3) / 4
    metrics("Hybrid × 3 + Phase 3 × 1", y_test, p_hyb_p3, results)

    p_all = (2 * p_hybrid + 2 * p_p3 + 1 * p_dc) / 5
    metrics("Hybrid × 2 + Phase 3 × 2 + DC × 1", y_test, p_all, results)

    # ── Summary
    print("\n" + "=" * 80)
    print("SUMMARY (now with no DC leakage)")
    print("=" * 80)
    rdf = pd.DataFrame(results).round(4)
    rdf = rdf[["name", "ll", "acc", "rps", "brier", "ece", "aw_r", "dr_r", "hw_r"]]
    print(rdf.to_string(index=False))
    rdf.to_csv(PROC / "phase3_clean_results.csv", index=False)

    # Save best model (whichever has lowest log loss)
    best_idx = rdf["ll"].idxmin()
    print(f"\n  Best by log loss: {rdf.iloc[best_idx]['name']} (ll={rdf.iloc[best_idx]['ll']})")


if __name__ == "__main__":
    main()
