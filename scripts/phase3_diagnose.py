"""
Phase 3 diagnostics — answer "do the new B5 features actually contribute?"

Two analyses:
  1. Feature importance from the trained XGBoost — which features does the
     model actually rely on? Rank top 30 + show how the new B5 features fare.
  2. Slice the holdout by match type:
       - friendlies vs competitive
       - qualifying vs major tournament
     Phase 3's new features were derived from international tournaments, so
     if they add value it should show on the *major tournament* subset more
     than the friendly/qualifying subset.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"

# Re-import the join logic from training to recompute test features
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    BASE_FEATURES, HOME_SQUAD, AWAY_SQUAD, DIFF_FEATURES,
    HOME_SB, AWAY_SB, HOME_INTL, AWAY_INTL, HOME_CHEM, AWAY_CHEM,
    NEW_DIFF_FEATURES, PHASE2_FEATURES, PHASE3_FEATURES,
    join_squad, join_static, dc_probs_fn, evaluate,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)


def feature_importance(model_path: Path):
    """Pull feature importance from the calibrated XGBClassifier inside the model."""
    bundle = joblib.load(model_path)
    cal = bundle["xgb"]  # CalibratedClassifierCV
    feature_cols = bundle["features_xgb"]

    # CalibratedClassifierCV wraps multiple folds. Each has a calibrated_classifiers_
    # entry, each of which has a .estimator (the actual XGBClassifier).
    importances = []
    for c in cal.calibrated_classifiers_:
        est = c.estimator
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    avg = np.mean(importances, axis=0)
    df = pd.DataFrame({"feature": feature_cols, "importance": avg})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def main():
    print("=" * 80)
    print("PHASE 3 DIAGNOSTICS")
    print("=" * 80)

    # ── Feature importance
    print("\n[1] Feature importance from trained XGBoost (Phase 3 model)")
    fi = feature_importance(MODELS / "phase3_model.pkl")
    print(f"\n  Top 20 features:")
    print(fi.head(20).to_string(index=False))

    # Categorize features
    new_b5 = HOME_SB + AWAY_SB + HOME_INTL + AWAY_INTL + HOME_CHEM + AWAY_CHEM + NEW_DIFF_FEATURES
    new_b5_set = set(new_b5)
    fi["category"] = fi["feature"].apply(
        lambda f: "B5_new" if f in new_b5_set else
                  "squad" if f in (HOME_SQUAD + AWAY_SQUAD + DIFF_FEATURES) else
                  "base"
    )
    summary = fi.groupby("category").agg(
        n=("feature", "count"),
        total_importance=("importance", "sum"),
        max_importance=("importance", "max"),
        mean_importance=("importance", "mean"),
    ).round(5)
    print(f"\n  Importance by category:")
    print(summary.to_string())

    # Where do the new B5 features rank?
    fi["rank"] = fi.index + 1
    b5_in_model = fi[fi["category"] == "B5_new"].copy()
    print(f"\n  New B5 features ranked (top 10 by importance):")
    print(b5_in_model.head(10)[["rank", "feature", "importance"]].to_string(index=False))
    print(f"\n  → Total B5 share of model: {summary.loc['B5_new','total_importance']*100:.1f}%")

    # ── Slice holdout
    print("\n" + "=" * 80)
    print("[2] Slice holdout by match type — do B5 features help on tournament matches?")
    print("=" * 80)

    train_dc = pd.read_csv(PROC / "train_dc.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    test = join_squad(test, tfy)
    test = join_static(test, sb, intl, chem)

    bundle = joblib.load(MODELS / "phase3_model.pkl")
    le = bundle["label_encoder"]
    y_test = le.transform(test["outcome"].values)

    # Phase 3 model predictions
    p_xgb_p3 = bundle["xgb"].predict_proba(test[bundle["features_xgb"]].values)

    # Need Phase 2-style XGB only: refit a small XGB on PHASE2_FEATURES from same train
    # Faster: just take the difference between Phase 3 and Phase 2 architectures by
    # running M0 features through the SAME Phase 3 XGB. Not perfect but quick.
    # Better: re-extract from earlier M0 results — but we didn't save M0 model.
    # Quick approach: re-train just M0 XGB here.

    print("\n  Refitting Phase 2-arch XGB for comparison...")
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb_
    XGB_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)

    train = pool[pool["date"] < HOLDOUT_START].copy()
    train = join_squad(train, tfy)
    train = join_static(train, sb, intl, chem)
    y_train = le.transform(train["outcome"].values)

    xgb_m0 = CalibratedClassifierCV(
        xgb_.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_m0.fit(train[PHASE2_FEATURES].values, y_train)
    p_xgb_p2 = xgb_m0.predict_proba(test[PHASE2_FEATURES].values)

    # Slice by match type
    slices = {
        "ALL": np.ones(len(test), dtype=bool),
        "Friendlies": (test["tournament_category"] == "friendly").values,
        "Qualifiers": (test["tournament_category"] == "qualifier").values,
        "Continental/Major": test["tournament_category"].isin(
            ["continental_final", "world_cup", "other_competitive"]).values,
    }

    rows = []
    print(f"\n{'Slice':<22} {'N':>4}  {'P2 ll':>7}  {'P3 ll':>7}  {'Δll':>7}  "
          f"{'P2 acc':>7}  {'P3 acc':>7}  {'Δacc':>7}")
    for sname, mask in slices.items():
        if mask.sum() < 10:
            continue
        ll_p2 = log_loss(y_test[mask], p_xgb_p2[mask])
        ll_p3 = log_loss(y_test[mask], p_xgb_p3[mask])
        acc_p2 = accuracy_score(y_test[mask], np.argmax(p_xgb_p2[mask], axis=1))
        acc_p3 = accuracy_score(y_test[mask], np.argmax(p_xgb_p3[mask], axis=1))
        d_ll = ll_p3 - ll_p2
        d_acc = acc_p3 - acc_p2
        rows.append({"slice": sname, "n": int(mask.sum()),
                     "p2_ll": ll_p2, "p3_ll": ll_p3, "delta_ll": d_ll,
                     "p2_acc": acc_p2, "p3_acc": acc_p3, "delta_acc": d_acc})
        arrow_ll = "↓" if d_ll < 0 else "↑"
        arrow_acc = "↑" if d_acc > 0 else "↓"
        print(f"{sname:<22} {mask.sum():>4}  {ll_p2:.4f}  {ll_p3:.4f}  "
              f"{arrow_ll}{abs(d_ll):.4f}  {acc_p2:.4f}  {acc_p3:.4f}  {arrow_acc}{abs(d_acc):.4f}")

    pd.DataFrame(rows).to_csv(PROC / "phase3_slice_analysis.csv", index=False)
    print(f"\n  Saved: data/processed/phase3_slice_analysis.csv")

    # ── Interpretation
    print("\n" + "=" * 80)
    print("[3] Interpretation")
    print("=" * 80)
    b5_total = summary.loc["B5_new", "total_importance"]
    base_total = summary.loc["base", "total_importance"]
    squad_total = summary.loc["squad", "total_importance"]
    print(f"  Feature importance breakdown:")
    print(f"    Base features (ELO, form, H2H, DC, conf): {base_total*100:.1f}%")
    print(f"    Squad features (FIFA ratings):            {squad_total*100:.1f}%")
    print(f"    NEW B5 features:                          {b5_total*100:.1f}%")
    if b5_total < 0.05:
        print(f"\n  → B5 features are <5% of model attention. They are nearly ignored.")
    elif b5_total < 0.15:
        print(f"\n  → B5 features modest signal — model uses them but they're not central.")
    else:
        print(f"\n  → B5 features are meaningful contributors.")


if __name__ == "__main__":
    main()
