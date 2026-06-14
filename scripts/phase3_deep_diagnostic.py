"""
Phase 3 deep diagnostic — stop reporting summary numbers, actually understand
where the model fails.

Outputs (docs/images/phase3/):
  01_calibration.png       reliability diagram per class (Phase 2 vs Phase 3)
  02_confusion_matrix.png  per-class accuracy + bias toward outcomes
  03_error_by_elo.png      log loss by ELO-difference bucket
  04_error_by_tournament.png  per tournament type
  05_pred_distribution.png  predicted probability histograms
  06_worst_misses.csv      20 worst-predicted matches in test holdout

Also prints comparison: where do P2 and P3 disagree? Who wins?
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
IMG = ROOT / "docs/images/phase3"
IMG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    BASE_FEATURES, HOME_SQUAD, AWAY_SQUAD, DIFF_FEATURES,
    HOME_SB, AWAY_SB, HOME_INTL, AWAY_INTL, HOME_CHEM, AWAY_CHEM,
    NEW_DIFF_FEATURES, PHASE2_FEATURES, PHASE3_FEATURES,
    join_squad, join_static,
    TRAIN_START, HOLDOUT_START, HOLDOUT_END,
)

CLASS_NAMES = ["away_win", "draw", "home_win"]
CLASS_COLORS = ["#E74C3C", "#F39C12", "#27AE60"]


def load_data():
    train_dc = pd.read_csv(PROC / "train_dc.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()
    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")

    for df_name in [(train, "train"), (test, "test")]:
        pass
    train = join_squad(train, tfy)
    train = join_static(train, sb, intl, chem)
    test = join_squad(test, tfy)
    test = join_static(test, sb, intl, chem)
    return train, test


def get_predictions(train, test):
    """Return (y_true, p2_proba, p3_proba) for the test set."""
    bundle = joblib.load(MODELS / "phase3_model.pkl")
    le = bundle["label_encoder"]
    y_test = le.transform(test["outcome"].values)

    XGB_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    y_train = le.transform(train["outcome"].values)

    # Phase 3 predictions (use saved model)
    p3 = bundle["xgb"].predict_proba(test[bundle["features_xgb"]].values)

    # Phase 2 predictions (refit XGB on PHASE2_FEATURES)
    print("  Refitting Phase 2-arch XGB...")
    xgb_p2 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_p2.fit(train[PHASE2_FEATURES].values, y_train)
    p2 = xgb_p2.predict_proba(test[PHASE2_FEATURES].values)

    return y_test, p2, p3


def plot_calibration(y_true, p2, p3, n_bins=10):
    """Reliability diagram per class. X = predicted prob bucket, Y = actual freq."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0, 1, n_bins + 1)

    for cls_idx, (name, color, ax) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, axes)):
        for model_p, label, marker in [(p2, "Phase 2", "o"), (p3, "Phase 3", "s")]:
            probs = model_p[:, cls_idx]
            actuals = (y_true == cls_idx).astype(int)
            bin_centers, bin_actual_rate, bin_count = [], [], []
            for i in range(n_bins):
                m = (probs >= bins[i]) & (probs < bins[i + 1])
                if m.sum() >= 5:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_actual_rate.append(actuals[m].mean())
                    bin_count.append(m.sum())
            ax.plot(bin_centers, bin_actual_rate, marker + "-",
                    label=f"{label} (n bins={len(bin_centers)})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.set_title(f"Class: {name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Calibration / Reliability Diagram — closer to diagonal = better calibrated",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(IMG / "01_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 01_calibration.png")


def plot_confusion_matrix(y_true, p2, p3):
    pred_p2 = np.argmax(p2, axis=1)
    pred_p3 = np.argmax(p3, axis=1)
    cm_p2 = confusion_matrix(y_true, pred_p2, normalize="true")
    cm_p3 = confusion_matrix(y_true, pred_p3, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, name in [(axes[0], cm_p2, "Phase 2"), (axes[1], cm_p3, "Phase 3")]:
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    cbar_kws={"label": "frequency"})
        ax.set_title(f"{name} — confusion (row-normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(IMG / "02_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 02_confusion_matrix.png")

    # Also print per-class precision/recall
    print("\n  Per-class diagonal accuracy (recall):")
    print(f"    Phase 2: {dict(zip(CLASS_NAMES, cm_p2.diagonal().round(3)))}")
    print(f"    Phase 3: {dict(zip(CLASS_NAMES, cm_p3.diagonal().round(3)))}")


def plot_error_by_elo(y_true, p2, p3, test):
    """Log loss by ELO-diff bucket."""
    test = test.copy().reset_index(drop=True)
    test["abs_elo_diff"] = test["elo_diff"].abs()
    buckets = pd.cut(test["abs_elo_diff"], bins=[0, 50, 100, 200, 400, 9999],
                     labels=["close (0-50)", "moderate (50-100)", "clear (100-200)",
                             "lopsided (200-400)", "blowout (400+)"])
    test["bucket"] = buckets

    rows = []
    for b in buckets.cat.categories:
        m = (test["bucket"] == b).values
        if m.sum() < 5:
            continue
        rows.append({
            "bucket": b, "n": int(m.sum()),
            "p2_ll": log_loss(y_true[m], p2[m], labels=[0, 1, 2]),
            "p3_ll": log_loss(y_true[m], p3[m], labels=[0, 1, 2]),
            "p2_acc": accuracy_score(y_true[m], np.argmax(p2[m], axis=1)),
            "p3_acc": accuracy_score(y_true[m], np.argmax(p3[m], axis=1)),
        })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w/2, df["p2_ll"], w, label="Phase 2", color="#3498DB", alpha=0.85)
    axes[0].bar(x + w/2, df["p3_ll"], w, label="Phase 3", color="#9B59B6", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["bucket"], rotation=15)
    axes[0].set_ylabel("Log loss (lower = better)")
    axes[0].set_title("Error by |ELO diff| — where do close matches hurt?")
    axes[0].legend()
    for i, n in enumerate(df["n"]):
        axes[0].text(i, max(df["p2_ll"][i], df["p3_ll"][i]) + 0.02, f"n={n}",
                     ha="center", fontsize=8, color="gray")

    axes[1].bar(x - w/2, df["p2_acc"], w, label="Phase 2", color="#3498DB", alpha=0.85)
    axes[1].bar(x + w/2, df["p3_acc"], w, label="Phase 3", color="#9B59B6", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["bucket"], rotation=15)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by |ELO diff|")
    axes[1].axhline(1/3, color="red", linestyle="--", alpha=0.5, label="random (1/3)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(IMG / "03_error_by_elo.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 03_error_by_elo.png")
    print("\n  Error by ELO diff bucket:")
    print(df.round(4).to_string(index=False))


def plot_error_by_tournament(y_true, p2, p3, test):
    test = test.copy().reset_index(drop=True)
    rows = []
    for cat in test["tournament_category"].unique():
        m = (test["tournament_category"] == cat).values
        if m.sum() < 10:
            continue
        rows.append({
            "tournament": cat, "n": int(m.sum()),
            "p2_ll": log_loss(y_true[m], p2[m], labels=[0, 1, 2]),
            "p3_ll": log_loss(y_true[m], p3[m], labels=[0, 1, 2]),
            "p2_acc": accuracy_score(y_true[m], np.argmax(p2[m], axis=1)),
            "p3_acc": accuracy_score(y_true[m], np.argmax(p3[m], axis=1)),
            "delta_acc": accuracy_score(y_true[m], np.argmax(p3[m], axis=1)) -
                         accuracy_score(y_true[m], np.argmax(p2[m], axis=1)),
        })
    df = pd.DataFrame(rows).sort_values("delta_acc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["p2_acc"], w, label="Phase 2", color="#3498DB", alpha=0.85)
    ax.bar(x + w/2, df["p3_acc"], w, label="Phase 3", color="#9B59B6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["tournament"], rotation=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by tournament category")
    for i, n in enumerate(df["n"]):
        ax.text(i, max(df["p2_acc"].iloc[i], df["p3_acc"].iloc[i]) + 0.01, f"n={n}",
                ha="center", fontsize=9, color="gray")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG / "04_error_by_tournament.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 04_error_by_tournament.png")
    print("\n  Accuracy by tournament:")
    print(df.round(4).to_string(index=False))


def plot_pred_distribution(y_true, p2, p3):
    """How does the model distribute its predictions? Are probs clustered or spread?"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, (model_p, label) in enumerate([(p2, "Phase 2"), (p3, "Phase 3")]):
        for col, (cls_idx, name, color) in enumerate(zip([0, 1, 2], CLASS_NAMES, CLASS_COLORS)):
            ax = axes[row][col]
            ax.hist(model_p[:, cls_idx], bins=30, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(1/3, color="red", linestyle="--", alpha=0.5, label="random (1/3)")
            ax.set_xlabel(f"Predicted P({name})")
            ax.set_ylabel("Count")
            ax.set_title(f"{label}: {name} probability distribution")
            ax.legend()
    plt.tight_layout()
    plt.savefig(IMG / "05_pred_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → 05_pred_distribution.png")


def save_worst_misses(y_true, p3, test):
    """The 20 matches where Phase 3 was most wrong."""
    test = test.copy().reset_index(drop=True)
    actual_prob = np.array([p3[i, y_true[i]] for i in range(len(y_true))])
    test["prob_actual"] = actual_prob
    test["pred_outcome"] = [CLASS_NAMES[i] for i in np.argmax(p3, axis=1)]
    test["pred_prob"] = np.max(p3, axis=1)
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "outcome", "pred_outcome", "pred_prob", "prob_actual",
            "elo_diff", "tournament_category"]
    worst = test.nsmallest(20, "prob_actual")[cols].copy()
    worst["prob_actual"] = worst["prob_actual"].round(3)
    worst["pred_prob"] = worst["pred_prob"].round(3)
    worst["elo_diff"] = worst["elo_diff"].round(0).astype(int)
    worst.to_csv(IMG / "06_worst_misses.csv", index=False)
    print("\n  Top 10 worst-predicted matches by Phase 3 (model gave actual outcome very low prob):")
    print(worst.head(10).to_string(index=False))


def disagreement_analysis(y_true, p2, p3, test):
    pred_p2 = np.argmax(p2, axis=1)
    pred_p3 = np.argmax(p3, axis=1)
    disagree = pred_p2 != pred_p3
    n_disagree = disagree.sum()
    print(f"\n  P2 vs P3 disagreement: {n_disagree}/{len(y_true)} matches ({n_disagree/len(y_true)*100:.1f}%)")
    if n_disagree == 0:
        return
    p2_right = (pred_p2[disagree] == y_true[disagree]).sum()
    p3_right = (pred_p3[disagree] == y_true[disagree]).sum()
    print(f"    When they disagree: P2 right {p2_right}, P3 right {p3_right} "
          f"({'Phase 3 wins' if p3_right > p2_right else 'Phase 2 wins' if p2_right > p3_right else 'tied'})")


def main():
    print("=" * 80)
    print("PHASE 3 DEEP DIAGNOSTIC")
    print("=" * 80)
    print("\n[Load]")
    train, test = load_data()
    print(f"  train: {len(train):,}, test: {len(test):,}")

    print("\n[Predict]")
    y_true, p2, p3 = get_predictions(train, test)
    print(f"  Phase 2 log loss: {log_loss(y_true, p2):.4f}, acc: {accuracy_score(y_true, np.argmax(p2,axis=1)):.4f}")
    print(f"  Phase 3 log loss: {log_loss(y_true, p3):.4f}, acc: {accuracy_score(y_true, np.argmax(p3,axis=1)):.4f}")

    print("\n[Plots]")
    plot_calibration(y_true, p2, p3)
    plot_confusion_matrix(y_true, p2, p3)
    plot_error_by_elo(y_true, p2, p3, test)
    plot_error_by_tournament(y_true, p2, p3, test)
    plot_pred_distribution(y_true, p2, p3)
    save_worst_misses(y_true, p3, test)
    disagreement_analysis(y_true, p2, p3, test)

    print(f"\nAll outputs → {IMG}")


if __name__ == "__main__":
    main()
