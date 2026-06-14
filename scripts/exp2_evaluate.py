"""
Experiment 2 — Evaluate models with new time-varying contextual features.

Compares:
  • Phase 3 baseline (current best: hybrid x3 + classifier x1 blend)
  • Phase 3 + 16 new time-varying features
  • Same architecture variants (hybrid alone, classifier alone, blend)

Reports on BOTH holdouts:
  • Full holdout (748 matches)
  • Tournament-only holdout (316 matches — what matters for WC 2026)

Also: slice metrics by close vs lopsided matches to see if the new features
help where Phase 3 collapsed (ELO diff < 100 → 37-46% accuracy).
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
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
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
TIMEVAR_FEATS = [
    "elo_momentum_6mo", "sos_adjusted_pts_10",
    "goals_vs_strong_10", "conceded_vs_strong_10",
    "streak_unbeaten", "volatility_gd_10",
    "tourn_xp_4y", "days_since_competitive",
]
HOME_TV = [f"home_{f}" for f in TIMEVAR_FEATS]
AWAY_TV = [f"away_{f}" for f in TIMEVAR_FEATS]
DIFF_TV = ["elo_momentum_diff", "sos_pts_diff", "streak_diff",
           "volatility_diff", "tourn_xp_diff"]
ALL_TV = HOME_TV + AWAY_TV + DIFF_TV
PHASE3_PLUS_TV = PHASE3_FEATURES + ALL_TV


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


def metrics(name, y_true, y_proba, results):
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, np.argmax(y_proba, axis=1))
    y_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))
    cum_p = np.cumsum(y_proba, axis=1); cum_t = np.cumsum(y_oh, axis=1)
    rps = float(np.mean(np.sum((cum_p - cum_t) ** 2, axis=1)) / 2)
    cm = confusion_matrix(y_true, np.argmax(y_proba, axis=1), normalize="true", labels=[0, 1, 2])
    aw_r, dr_r, hw_r = cm.diagonal()
    results.append({"name": name, "ll": ll, "acc": acc, "brier": brier, "rps": rps,
                    "aw_r": aw_r, "dr_r": dr_r, "hw_r": hw_r})


def join_timevar(df, tv):
    tv = tv.copy()
    tv["date"] = pd.to_datetime(tv["date"])
    # matches_clean has 171 duplicate match keys (multi-source ingest) — dedupe
    tv = tv.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    df = df.merge(tv, on=["date", "home_team", "away_team"], how="left")
    # Diff features
    df["elo_momentum_diff"] = df["home_elo_momentum_6mo"] - df["away_elo_momentum_6mo"]
    df["sos_pts_diff"] = df["home_sos_adjusted_pts_10"] - df["away_sos_adjusted_pts_10"]
    df["streak_diff"] = df["home_streak_unbeaten"] - df["away_streak_unbeaten"]
    df["volatility_diff"] = df["home_volatility_gd_10"] - df["away_volatility_gd_10"]
    df["tourn_xp_diff"] = df["home_tourn_xp_4y"] - df["away_tourn_xp_4y"]
    return df


def train_models(train, test, features, label):
    XGB_REG_P = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                     objective="count:poisson", eval_metric="poisson-nloglik",
                     random_state=42, verbosity=0)
    XGB_CLS_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                     eval_metric="mlogloss", random_state=42, verbosity=0)

    X_train = train[features].values
    X_test = test[features].values

    le = LabelEncoder()
    le.fit(["away_win", "draw", "home_win"])
    y_train = le.transform(train["outcome"].values)
    y_test = le.transform(test["outcome"].values)
    y_tr_h = train["home_score"].astype(int).values
    y_tr_a = train["away_score"].astype(int).values

    mh = xgb.XGBRegressor(**XGB_REG_P)
    mh.fit(X_train, y_tr_h)
    ma = xgb.XGBRegressor(**XGB_REG_P)
    ma.fit(X_train, y_tr_a)
    lh_tr = mh.predict(X_train); lh_te = mh.predict(X_test)
    la_tr = ma.predict(X_train); la_te = ma.predict(X_test)
    rho = fit_rho(y_tr_h, y_tr_a, lh_tr, la_tr)
    p_hyb = batch_wdl(lh_te, la_te, rho)

    cls = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_CLS_P), method="isotonic", cv=5)
    cls.fit(X_train, y_train)
    p_cls = cls.predict_proba(X_test)

    p_blend = (3 * p_hyb + 1 * p_cls) / 4
    return {"hybrid": p_hyb, "classifier": p_cls, "blend": p_blend,
            "y_test": y_test, "label": label, "features": features,
            "models": {"home_reg": mh, "away_reg": ma, "rho": rho, "classifier": cls}}


def main():
    print("=" * 90)
    print("EXPERIMENT 2 — evaluate with time-varying contextual features")
    print("=" * 90)

    # Load all data
    train_dc = pd.read_csv(PROC / "train_dc_v2.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc_v2.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    tv = pd.read_csv(PROC / "timevar_match_features.csv")

    pool = join_squad(pool, tfy)
    pool = join_static(pool, sb, intl, chem)
    pool = join_timevar(pool, tv)

    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()
    test_tourn = test[test["tournament_category"].isin(
        ["continental_final", "world_cup", "other_competitive"])].copy()

    print(f"\n  train: {len(train):,} (with TV features), test: {len(test):,}, "
          f"test_tournament: {len(test_tourn):,}")
    tv_coverage = train[HOME_TV].notna().all(axis=1).mean()
    print(f"  full TV-feature row coverage in train: {tv_coverage*100:.1f}%")

    # ── A: Phase 3 baseline (no TV)
    print("\n[A] Phase 3 baseline (no TV features)...")
    A = train_models(train, test, PHASE3_FEATURES, "Phase3")
    A_t = train_models(train, test_tourn, PHASE3_FEATURES, "Phase3 [tourn]")

    # ── B: Phase 3 + TV
    print("[B] Phase 3 + time-varying features...")
    B = train_models(train, test, PHASE3_PLUS_TV, "Phase3+TV")
    B_t = train_models(train, test_tourn, PHASE3_PLUS_TV, "Phase3+TV [tourn]")

    # ── Print
    print("\n" + "=" * 90)
    print("FULL HOLDOUT (748 matches)")
    print("=" * 90)
    res = []
    for run in [A, B]:
        for arch in ["hybrid", "classifier", "blend"]:
            metrics(f"{run['label']:<12} | {arch:<10}", run["y_test"], run[arch], res)
    rdf = pd.DataFrame(res)
    print(rdf.to_string(index=False))

    print("\n" + "=" * 90)
    print("TOURNAMENT-ONLY HOLDOUT (316 matches)")
    print("=" * 90)
    res_t = []
    for run in [A_t, B_t]:
        for arch in ["hybrid", "classifier", "blend"]:
            metrics(f"{run['label']:<20} | {arch:<10}", run["y_test"], run[arch], res_t)
    rdf_t = pd.DataFrame(res_t)
    print(rdf_t.to_string(index=False))

    # ── Slice by ELO diff bucket — does TV help close matches?
    print("\n" + "=" * 90)
    print("SLICE: full holdout by |ELO diff| bucket")
    print("=" * 90)
    abs_elo = test["elo_diff"].abs().values
    buckets = pd.cut(abs_elo, bins=[0, 50, 100, 200, 400, 9999],
                     labels=["close (0-50)", "moderate (50-100)",
                             "clear (100-200)", "lopsided (200-400)", "blowout (400+)"])
    buckets = pd.Series(buckets).reset_index(drop=True)

    print(f"{'Bucket':<22} {'n':>4}  {'Phase3 ll':>10}  {'Phase3+TV ll':>14}  "
          f"{'Δ ll':>7}  {'P3 acc':>7}  {'P3+TV acc':>10}  {'Δ acc':>7}")
    for b in buckets.cat.categories:
        m = (buckets == b).values
        if m.sum() < 5:
            continue
        ll_p3 = log_loss(A["y_test"][m], A["blend"][m], labels=[0, 1, 2])
        ll_tv = log_loss(B["y_test"][m], B["blend"][m], labels=[0, 1, 2])
        acc_p3 = accuracy_score(A["y_test"][m], np.argmax(A["blend"][m], axis=1))
        acc_tv = accuracy_score(B["y_test"][m], np.argmax(B["blend"][m], axis=1))
        d_ll = ll_tv - ll_p3
        d_acc = acc_tv - acc_p3
        arrow_ll = "↓" if d_ll < 0 else "↑"
        arrow_acc = "↑" if d_acc > 0 else "↓"
        print(f"{str(b):<22} {m.sum():>4}  {ll_p3:.4f}      {ll_tv:.4f}        "
              f"{arrow_ll}{abs(d_ll):.4f}  {acc_p3:.4f}  {acc_tv:.4f}      {arrow_acc}{abs(d_acc):.4f}")

    # ── Feature importance for the TV features specifically
    print("\n" + "=" * 90)
    print("Feature importance of new TV features (in Phase3+TV hybrid home-goals regressor)")
    print("=" * 90)
    mh = B["models"]["home_reg"]
    fi = pd.DataFrame({
        "feature": PHASE3_PLUS_TV,
        "importance": mh.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    tv_set = set(HOME_TV + AWAY_TV + DIFF_TV)
    tv_in_top = fi[fi["feature"].isin(tv_set)].head(15)
    print(tv_in_top.to_string(index=False))
    print(f"\n  Total TV share of model importance: {fi[fi['feature'].isin(tv_set)]['importance'].sum()*100:.1f}%")

    print("\n  ALL features ranked top 20:")
    print(fi.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
