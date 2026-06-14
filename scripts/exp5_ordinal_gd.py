"""
Experiment 5 — Ordinal regression on goal difference.

Hypothesis: the 3-way W/D/L target destroys information. A 5-0 home win
trains the same way as a 1-0 home win, and the model can't distinguish
"these teams are evenly matched" (draw zone) from "home is dominant"
(big-win zone). Reformulating into ordered goal-diff buckets:

  Bucket: -3+ | -2 | -1 | 0 | +1 | +2 | +3+
  W/D/L = sum(bucket≤-1)   bucket=0    sum(bucket≥+1)

Draws emerge naturally because the model learns that close matches
produce goal-diff distributions centred on 0, not bimodal at ±1.

Three approaches tested:
  A. XGB multiclass on goal-diff buckets (k=7)
  B. XGB multiclass on goal-diff buckets (k=11, finer)
  C. Cumulative logistic regression (proper ordinal) via mord library

Compared against:
  • Phase 3 classifier (3-way XGB)
  • Phase 3 hybrid (Poisson)
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

# Try to import mord for proportional-odds ordinal
try:
    from mord import LogisticIT
    HAS_MORD = True
except ImportError:
    HAS_MORD = False


def bucket_gd(gd: int, max_abs: int = 3) -> int:
    """Bucket goal-diff to [-max_abs, +max_abs] with clipping.
    Returns class index 0..2*max_abs (so 0..6 for max_abs=3)."""
    return int(np.clip(gd, -max_abs, max_abs)) + max_abs


def gd_buckets_to_wdl(probs: np.ndarray, max_abs: int) -> np.ndarray:
    """Convert (n, 2*max_abs+1) gd-bucket probs → (n, 3) W/D/L probs.
    Class order: [away_win, draw, home_win]."""
    away_idx = list(range(max_abs))           # buckets [-max_abs, ..., -1]
    draw_idx = max_abs                        # bucket 0
    home_idx = list(range(max_abs + 1, 2 * max_abs + 1))  # [+1, ..., +max_abs]
    out = np.zeros((len(probs), 3))
    out[:, 0] = probs[:, away_idx].sum(axis=1)
    out[:, 1] = probs[:, draw_idx]
    out[:, 2] = probs[:, home_idx].sum(axis=1)
    out = out / out.sum(axis=1, keepdims=True)
    return out


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
    print(f"  {name:50s}  ll={ll:.4f}  acc={acc:.4f}  RPS={rps:.4f}  draw_r={dr_r:.3f}")


def main():
    print("=" * 90)
    print("EXPERIMENT 5 — Ordinal regression on goal difference")
    print("=" * 90)

    # Load + join everything (same setup as exp2)
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
    test_tourn = test[test["tournament_category"].isin(
        ["continental_final", "world_cup", "other_competitive"])].copy()

    print(f"\n  train: {len(train):,}, test: {len(test):,}, tournament: {len(test_tourn):,}")

    # Encode targets
    le = LabelEncoder()
    le.fit(["away_win", "draw", "home_win"])
    y_train_wdl = le.transform(train["outcome"].values)
    y_test_wdl = le.transform(test["outcome"].values)
    y_test_t_wdl = le.transform(test_tourn["outcome"].values)

    # Goal diff
    gd_train = (train["home_score"] - train["away_score"]).astype(int).values
    gd_test = (test["home_score"] - test["away_score"]).astype(int).values
    gd_test_t = (test_tourn["home_score"] - test_tourn["away_score"]).astype(int).values

    X_train = train[PHASE3_FEATURES].values
    X_test = test[PHASE3_FEATURES].values
    X_test_t = test_tourn[PHASE3_FEATURES].values

    print(f"\n  Goal-diff distribution (train):")
    s = pd.Series(gd_train).describe()
    print(f"    range: {int(gd_train.min())} → {int(gd_train.max())}, "
          f"mean: {gd_train.mean():.2f}, median: {int(np.median(gd_train))}")
    vc = pd.Series(gd_train).value_counts().sort_index()
    common = vc[vc.index.isin([-3, -2, -1, 0, 1, 2, 3])]
    print(f"    counts: {dict(common)}")

    results = []

    # ── A: XGB multiclass on goal-diff buckets, max_abs=3 (7 classes)
    print("\n" + "=" * 90)
    print("[A] XGB multiclass on goal-diff buckets, max_abs=3 (7 classes)")
    print("=" * 90)
    max_abs_a = 3
    y_train_gd = np.array([bucket_gd(g, max_abs_a) for g in gd_train])
    n_classes = 2 * max_abs_a + 1
    print(f"  classes: 0..{n_classes-1}, bucket distribution: {np.bincount(y_train_gd, minlength=n_classes).tolist()}")

    XGB_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                 objective="multi:softprob", num_class=n_classes,
                 eval_metric="mlogloss", random_state=42, verbosity=0)
    cls_A = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P), method="isotonic", cv=5)
    cls_A.fit(X_train, y_train_gd)
    p_gd_A = cls_A.predict_proba(X_test)
    p_wdl_A = gd_buckets_to_wdl(p_gd_A, max_abs_a)
    metrics("A: ordinal-gd k=7 (full holdout)", y_test_wdl, p_wdl_A, results)

    p_gd_A_t = cls_A.predict_proba(X_test_t)
    p_wdl_A_t = gd_buckets_to_wdl(p_gd_A_t, max_abs_a)
    metrics("A: ordinal-gd k=7 (tournament)", y_test_t_wdl, p_wdl_A_t, results)

    # ── B: XGB multiclass on goal-diff buckets, max_abs=5 (11 classes)
    print("\n" + "=" * 90)
    print("[B] XGB multiclass on goal-diff buckets, max_abs=5 (11 classes, finer)")
    print("=" * 90)
    max_abs_b = 5
    y_train_gd_b = np.array([bucket_gd(g, max_abs_b) for g in gd_train])
    n_classes_b = 2 * max_abs_b + 1
    print(f"  bucket distribution: {np.bincount(y_train_gd_b, minlength=n_classes_b).tolist()}")

    XGB_P_b = {**XGB_P, "num_class": n_classes_b}
    cls_B = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P_b), method="isotonic", cv=5)
    cls_B.fit(X_train, y_train_gd_b)
    p_gd_B = cls_B.predict_proba(X_test)
    p_wdl_B = gd_buckets_to_wdl(p_gd_B, max_abs_b)
    metrics("B: ordinal-gd k=11 (full holdout)", y_test_wdl, p_wdl_B, results)
    p_gd_B_t = cls_B.predict_proba(X_test_t)
    p_wdl_B_t = gd_buckets_to_wdl(p_gd_B_t, max_abs_b)
    metrics("B: ordinal-gd k=11 (tournament)", y_test_t_wdl, p_wdl_B_t, results)

    # ── C: Proper ordinal logistic regression (mord) — if available
    if HAS_MORD:
        print("\n" + "=" * 90)
        print("[C] Proportional-odds logistic regression (mord.LogisticIT)")
        print("=" * 90)
        # mord needs no NaN
        X_train_f = pd.DataFrame(X_train, columns=PHASE3_FEATURES).fillna(0).values
        X_test_f = pd.DataFrame(X_test, columns=PHASE3_FEATURES).fillna(0).values
        X_test_t_f = pd.DataFrame(X_test_t, columns=PHASE3_FEATURES).fillna(0).values

        mord_model = LogisticIT(alpha=1.0)
        # mord expects classes starting at 0, contiguous
        y_train_gd_b_contig = y_train_gd_b
        mord_model.fit(X_train_f, y_train_gd_b_contig)
        # mord's predict_proba (it has one)
        if hasattr(mord_model, "predict_proba"):
            p_gd_C = mord_model.predict_proba(X_test_f)
            p_wdl_C = gd_buckets_to_wdl(p_gd_C, max_abs_b)
            metrics("C: mord LogisticIT (full holdout)", y_test_wdl, p_wdl_C, results)
            p_gd_C_t = mord_model.predict_proba(X_test_t_f)
            p_wdl_C_t = gd_buckets_to_wdl(p_gd_C_t, max_abs_b)
            metrics("C: mord LogisticIT (tournament)", y_test_t_wdl, p_wdl_C_t, results)
        else:
            print("  mord LogisticIT has no predict_proba, skipping")
    else:
        print("\n[C] mord not installed — skip")

    # ── For comparison: Phase 3 baseline (3-way classifier + hybrid)
    print("\n" + "=" * 90)
    print("[BASELINE] Phase 3 3-way classifier (for comparison)")
    print("=" * 90)
    XGB_3WAY = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    eval_metric="mlogloss", random_state=42, verbosity=0)
    cls_3way = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_3WAY), method="isotonic", cv=5)
    cls_3way.fit(X_train, y_train_wdl)
    p_3way = cls_3way.predict_proba(X_test)
    p_3way_t = cls_3way.predict_proba(X_test_t)
    metrics("Phase3 3-way classifier (full)", y_test_wdl, p_3way, results)
    metrics("Phase3 3-way classifier (tournament)", y_test_t_wdl, p_3way_t, results)

    # ── Blend: best ordinal + 3-way classifier
    print("\n" + "=" * 90)
    print("[BLEND] best ordinal + 3-way classifier")
    print("=" * 90)
    # use B (k=11) since it's the richer ordinal
    p_blend = (3 * p_wdl_B + 1 * p_3way) / 4
    p_blend_t = (3 * p_wdl_B_t + 1 * p_3way_t) / 4
    metrics("B×3 + 3way×1 (full)", y_test_wdl, p_blend, results)
    metrics("B×3 + 3way×1 (tournament)", y_test_t_wdl, p_blend_t, results)

    # ── Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    rdf = pd.DataFrame(results).round(4)
    print(rdf.to_string(index=False))
    rdf.to_csv(PROC / "exp5_ordinal_results.csv", index=False)


if __name__ == "__main__":
    main()
