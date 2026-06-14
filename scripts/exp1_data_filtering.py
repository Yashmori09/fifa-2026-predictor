"""
Experiment 1 — Aggressive training-data filtering.

Hypothesis: ~40% of our 6,162 training matches are friendlies (B-teams,
rotation, meaningless results) or qualifier matches between very weak teams
(Andorra-Liechtenstein). These dilute the signal — the model has to fit one
set of weights to "friendly" and "WC knockout" simultaneously.

We test 4 filter strategies and measure the impact on log-loss/accuracy/RPS
of the same hybrid model architecture.

Holdout is reported TWO ways:
  • FULL holdout (all 748 matches June 2025 → March 2026)
  • TOURNAMENT-ONLY holdout (continental_final + WC + competitive non-friendly)
The tournament-only number is what matters for actual WC 2026 prediction.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    PHASE3_FEATURES, join_squad, join_static,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)

CLASS_NAMES = ["away_win", "draw", "home_win"]
MAX_GOALS = 10
XGB_REG_P = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                 objective="count:poisson", eval_metric="poisson-nloglik",
                 random_state=42, verbosity=0)
XGB_CLS_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                 eval_metric="mlogloss", random_state=42, verbosity=0)


def lams_to_wdl(lh, la, rho):
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= max(1 - lh * la * rho, 1e-10)
    M[0, 1] *= max(1 + lh * rho, 1e-10)
    M[1, 0] *= max(1 + la * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    hw = float(np.sum(M * (g[:, None] > g[None, :])))
    dr = float(np.sum(M * (g[:, None] == g[None, :])))
    return np.array([max(0, 1 - hw - dr), dr, hw])


def batch_wdl(lhs, las, rho):
    out = np.zeros((len(lhs), 3))
    for i in range(len(lhs)):
        out[i] = lams_to_wdl(lhs[i], las[i], rho)
    return out


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


def metrics_dict(y_true, y_proba):
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, np.argmax(y_proba, axis=1))
    y_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))
    cum_p = np.cumsum(y_proba, axis=1)
    cum_t = np.cumsum(y_oh, axis=1)
    rps = float(np.mean(np.sum((cum_p - cum_t) ** 2, axis=1)) / 2)
    cm = confusion_matrix(y_true, np.argmax(y_proba, axis=1), normalize="true", labels=[0, 1, 2])
    return {"ll": ll, "acc": acc, "brier": brier, "rps": rps,
            "aw_r": cm[0, 0], "dr_r": cm[1, 1], "hw_r": cm[2, 2]}


def train_and_eval(train_df, test_df, name):
    """Train hybrid + classifier, return metrics dict."""
    X_train = train_df[PHASE3_FEATURES].values
    X_test = test_df[PHASE3_FEATURES].values

    le = LabelEncoder()
    le.fit(["away_win", "draw", "home_win"])
    y_train = le.transform(train_df["outcome"].values)
    y_test = le.transform(test_df["outcome"].values)
    y_tr_h = train_df["home_score"].astype(int).values
    y_tr_a = train_df["away_score"].astype(int).values

    # Hybrid: two Poisson regressors
    mh = xgb.XGBRegressor(**XGB_REG_P)
    mh.fit(X_train, y_tr_h)
    ma = xgb.XGBRegressor(**XGB_REG_P)
    ma.fit(X_train, y_tr_a)
    lh_tr = mh.predict(X_train); lh_te = mh.predict(X_test)
    la_tr = ma.predict(X_train); la_te = ma.predict(X_test)
    rho = fit_rho(y_tr_h, y_tr_a, lh_tr, la_tr)
    p_hyb = batch_wdl(lh_te, la_te, rho)

    # 3-way classifier
    cls = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_CLS_P), method="isotonic", cv=5)
    cls.fit(X_train, y_train)
    p_cls = cls.predict_proba(X_test)

    # Blend
    p_blend = (3 * p_hyb + 1 * p_cls) / 4

    return {
        "n_train": len(train_df),
        "hybrid": metrics_dict(y_test, p_hyb),
        "classifier": metrics_dict(y_test, p_cls),
        "blend": metrics_dict(y_test, p_blend),
    }


def main():
    print("=" * 90)
    print("EXPERIMENT 1 — Training data filtering")
    print("=" * 90)

    # ── Load
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

    base_train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()

    # Tournament-only test subset
    test_tourn = test[test["tournament_category"].isin(
        ["continental_final", "world_cup", "other_competitive"])].copy()
    print(f"\n  Full holdout: {len(test)} matches")
    print(f"  Tournament-only holdout: {len(test_tourn)} matches")
    print(f"  Full training pool: {len(base_train)} matches")
    print(f"    by category: {base_train['tournament_category'].value_counts().to_dict()}")
    print(f"    avg ELO of involved teams: home={base_train['home_elo_before'].mean():.0f}, "
          f"away={base_train['away_elo_before'].mean():.0f}")

    # ── Define filter strategies
    def f_baseline(df): return df.copy()
    def f_no_friendly(df): return df[df["tournament_category"] != "friendly"].copy()
    def f_no_weak(df):
        return df[(df["home_elo_before"] >= 1500) & (df["away_elo_before"] >= 1500)].copy()
    def f_modern(df): return df[df["date"] >= "2020-01-01"].copy()
    def f_combined(df):
        return df[(df["tournament_category"] != "friendly") &
                  (df["home_elo_before"] >= 1500) & (df["away_elo_before"] >= 1500)].copy()
    def f_combined_modern(df):
        return df[(df["tournament_category"] != "friendly") &
                  (df["home_elo_before"] >= 1500) & (df["away_elo_before"] >= 1500) &
                  (df["date"] >= "2020-01-01")].copy()
    def f_tournament_only(df):
        return df[df["tournament_category"].isin(
            ["continental_final", "world_cup", "other_competitive"])].copy()
    def f_tournament_plus_qual(df):
        return df[(df["tournament_category"] != "friendly") &
                  (df["home_elo_before"] >= 1400) & (df["away_elo_before"] >= 1400)].copy()

    strategies = [
        ("S0: baseline (all matches)", f_baseline),
        ("S1: drop friendlies", f_no_friendly),
        ("S2: drop weak teams (ELO<1500)", f_no_weak),
        ("S3: modern era only (2020+)", f_modern),
        ("S4: drop friendlies + weak", f_combined),
        ("S5: drop friendlies + weak + 2020+", f_combined_modern),
        ("S6: tournament matches only", f_tournament_only),
        ("S7: drop friendlies + ELO≥1400", f_tournament_plus_qual),
    ]

    # ── Run each strategy
    all_results = []
    for sname, filt in strategies:
        train_filtered = filt(base_train)
        if len(train_filtered) < 200:
            print(f"\n  {sname}: TOO FEW SAMPLES ({len(train_filtered)}), skipping")
            continue
        print(f"\n  {sname}: train n={len(train_filtered)}")
        print(f"    composition: {train_filtered['tournament_category'].value_counts().to_dict()}")

        # eval on both holdouts
        full_res = train_and_eval(train_filtered, test, sname)
        tourn_res = train_and_eval(train_filtered, test_tourn, sname + " [tourn]")
        all_results.append({"name": sname, "n_train": len(train_filtered),
                            "full": full_res, "tourn": tourn_res})

    # ── Pretty-print summary
    print("\n" + "=" * 110)
    print("RESULTS — FULL holdout (748 matches: friendlies, qualifiers, tournaments)")
    print("=" * 110)
    print(f"{'Strategy':<40} {'N_tr':>5}  {'arch':<10} {'ll':>6} {'acc':>6} {'rps':>6} {'dr_r':>6}")
    for r in all_results:
        for arch in ["hybrid", "classifier", "blend"]:
            m = r["full"][arch]
            print(f"{r['name']:<40} {r['n_train']:>5}  {arch:<10} "
                  f"{m['ll']:.4f} {m['acc']:.4f} {m['rps']:.4f} {m['dr_r']:.3f}")
        print()

    print("\n" + "=" * 110)
    print(f"RESULTS — TOURNAMENT-ONLY holdout ({len(test_tourn)} matches: the kind we care about)")
    print("=" * 110)
    print(f"{'Strategy':<40} {'N_tr':>5}  {'arch':<10} {'ll':>6} {'acc':>6} {'rps':>6} {'dr_r':>6}")
    for r in all_results:
        for arch in ["hybrid", "classifier", "blend"]:
            m = r["tourn"][arch]
            print(f"{r['name']:<40} {r['n_train']:>5}  {arch:<10} "
                  f"{m['ll']:.4f} {m['acc']:.4f} {m['rps']:.4f} {m['dr_r']:.3f}")
        print()

    # Best hybrid by tournament log loss
    print("\n" + "=" * 110)
    print("Ranked: best HYBRID log loss on tournament holdout")
    print("=" * 110)
    ranked = sorted(all_results, key=lambda r: r["tourn"]["hybrid"]["ll"])
    for i, r in enumerate(ranked[:5]):
        m = r["tourn"]["hybrid"]
        print(f"  #{i+1}: {r['name']:<40} ll={m['ll']:.4f} acc={m['acc']:.4f} (n_train={r['n_train']})")


if __name__ == "__main__":
    main()
