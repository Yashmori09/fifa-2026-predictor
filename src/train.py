"""
FIFA 2026 Match Outcome Predictor — AutoResearch train.py
Target metric: minimize log_loss on test set (3-class: away_win / draw / home_win)
Baseline (Ensemble XGB+RF+LR, 52 features): log_loss = 0.8333, accuracy = 0.6212, f1_macro = 0.5313
Current best (59 features incl. DC): log_loss = 0.8131, accuracy = 0.6266, f1_macro = 0.5045
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("../data/processed")
MODELS_DIR    = Path("../models")

# ── Load data — DC-augmented CSVs (59 features) ───────────────────────────────
# train_dc.csv / test_dc.csv are saved by notebook 06 and include all 7 DC features.
# Fall back to plain train/test if DC files not found.
_train_dc = PROCESSED_DIR / "train_dc.csv"
_test_dc  = PROCESSED_DIR / "test_dc.csv"
if _train_dc.exists() and _test_dc.exists():
    train_df = pd.read_csv(_train_dc)
    test_df  = pd.read_csv(_test_dc)
    _has_dc  = True
else:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    _has_dc  = False

FEATURE_COLS = [
    "home_elo_before", "away_elo_before", "elo_diff",
    "home_win_rate_5", "home_avg_scored_5", "home_avg_conceded_5",
    "home_pts_per_match_5", "home_matches_played_5",
    "home_win_rate_10", "home_avg_scored_10", "home_avg_conceded_10",
    "home_pts_per_match_10", "home_matches_played_10",
    "away_win_rate_5", "away_avg_scored_5", "away_avg_conceded_5",
    "away_pts_per_match_5", "away_matches_played_5",
    "away_win_rate_10", "away_avg_scored_10", "away_avg_conceded_10",
    "away_pts_per_match_10", "away_matches_played_10",
    "h2h_home_win_rate", "h2h_home_avg_scored", "h2h_home_avg_conceded",
    "h2h_total_meetings", "h2h_recent_win_rate",
    "neutral.1", "tournament_importance",
    "home_conf_UEFA", "home_conf_CAF", "home_conf_AFC",
    "home_conf_CONCACAF", "home_conf_CONMEBOL", "home_conf_OFC", "home_conf_UNKNOWN",
    "away_conf_UEFA", "away_conf_CAF", "away_conf_AFC",
    "away_conf_CONCACAF", "away_conf_CONMEBOL", "away_conf_OFC", "away_conf_UNKNOWN",
    "same_confederation",
]

TARGET = "outcome"

def add_engineered_features(df):
    d = df.copy()
    # Non-linear ELO dominance
    d["elo_diff_sq"] = d["elo_diff"] ** 2 * np.sign(d["elo_diff"])
    # Form momentum (recent vs longer window)
    d["home_form_momentum"] = d["home_win_rate_5"] - d["home_win_rate_10"]
    d["away_form_momentum"] = d["away_win_rate_5"] - d["away_win_rate_10"]
    # Goal difference form
    d["home_goal_diff_form"] = d["home_avg_scored_5"] - d["home_avg_conceded_5"]
    d["away_goal_diff_form"] = d["away_avg_scored_5"] - d["away_avg_conceded_5"]
    d["net_goal_diff"] = d["home_goal_diff_form"] - d["away_goal_diff_form"]
    # Confidence-weighted H2H
    d["h2h_confidence"] = d["h2h_recent_win_rate"] * (d["h2h_total_meetings"] / (d["h2h_total_meetings"] + 5))
    # EXP-20: Draw-specific features
    # draw_rate = pts_per_match - 3*win_rate (exact derivation: pts = 3W + 1D, so D = pts - 3W)
    d["home_draw_rate_10"] = d["home_pts_per_match_10"] - 3 * d["home_win_rate_10"]
    d["away_draw_rate_10"] = d["away_pts_per_match_10"] - 3 * d["away_win_rate_10"]
    d["combined_draw_tendency"] = d["home_draw_rate_10"] + d["away_draw_rate_10"]
    # abs_elo_diff: small gap = evenly matched = draw more likely
    d["abs_elo_diff"] = np.abs(d["elo_diff"])
    # elo_balance: 1 = perfectly even, 0 = massive mismatch (sigmoid-like scaling)
    d["elo_balance"] = 1 / (1 + d["abs_elo_diff"] / 200)
    # dc_draw_dominance: how much DC favors draw over the best win outcome
    if "dc_draw_prob" in d.columns:
        d["dc_draw_dominance"] = d["dc_draw_prob"] - np.maximum(
            d["dc_home_win_prob"], d["dc_away_win_prob"]
        )
    return d

train_eng = add_engineered_features(train_df)
test_eng  = add_engineered_features(test_df)

ENG_COLS = FEATURE_COLS + [
    "elo_diff_sq", "home_form_momentum", "away_form_momentum",
    "home_goal_diff_form", "away_goal_diff_form", "net_goal_diff", "h2h_confidence",
]

# Add DC features if available (7 more features → 59 total)
DC_COLS = [
    "dc_home_win_prob", "dc_draw_prob", "dc_away_win_prob",
    "dc_lambda", "dc_mu", "dc_total_goals", "dc_goal_diff",
]
if _has_dc and all(c in train_df.columns for c in DC_COLS):
    ENG_COLS = ENG_COLS + DC_COLS
    print(f"Using 59 features (52 base + 7 DC)")
else:
    print(f"Using 52 features (DC features not found)")

X_train = train_eng[ENG_COLS].values
y_train_raw = train_df[TARGET].values
X_test  = test_eng[ENG_COLS].values
y_test_raw  = test_df[TARGET].values

# ── Encode target ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

# ── Class weights ──────────────────────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
# Map to encoded integer labels (0=away_win, 1=draw, 2=home_win)
class_weight_str = dict(zip(classes, weights))

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION — AutoResearch will modify this section
# ══════════════════════════════════════════════════════════════════════════════

# Logistic Regression
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight=class_weight_str,
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42,
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight=class_weight_str,
    random_state=42,
    n_jobs=-1,
)

# XGBoost — wrapped with isotonic calibration to fix overconfident probabilities
# EXP-24: n=500, lr=0.03 (more trees + lower LR consistently best on 59-feature set)
xgb_base = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)
xgb_model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)

# RF — wrapped with isotonic calibration (RF outputs hard probs near 0/1)
rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight=class_weight_str,
    random_state=42,
    n_jobs=-1,
)
rf = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE
# EXP-23: DC as 4th voter — blend XGB×4 + RF×1 + DC×5 (no LR)
# DC probabilities add orthogonal Poisson-based signal not captured by trees
# ══════════════════════════════════════════════════════════════════════════════

# Fit XGB and RF individually
xgb_model.fit(X_train_s, y_train)
rf.fit(X_train_s, y_train)

xgb_proba = xgb_model.predict_proba(X_test_s)
rf_proba  = rf.predict_proba(X_test_s)

# DC probabilities — LabelEncoder order: 0=away_win, 1=draw, 2=home_win
dc_proba_test = np.column_stack([
    test_eng["dc_away_win_prob"].values,
    test_eng["dc_draw_prob"].values,
    test_eng["dc_home_win_prob"].values,
])
dc_proba_test = dc_proba_test / dc_proba_test.sum(axis=1, keepdims=True)

# Blend: XGB×4 + RF×1 + DC×5
W_XGB, W_RF, W_DC = 4, 1, 5
y_pred_proba = (W_XGB * xgb_proba + W_RF * rf_proba + W_DC * dc_proba_test) / (W_XGB + W_RF + W_DC)
y_pred       = np.argmax(y_pred_proba, axis=1)

acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="macro")
ll     = log_loss(y_test, y_pred_proba)

# ── Output — AutoResearch reads this ──────────────────────────────────────────
print(f"accuracy: {acc:.4f}")
print(f"f1_macro: {f1:.4f}")
print(f"log_loss: {ll:.4f}")

# AutoResearch target line (it parses "metric: value" format)
print(f"METRIC: {ll:.4f}")

# ── Save best model ────────────────────────────────────────────────────────────
import joblib, json
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump({"xgb": xgb_model, "rf": rf, "w_xgb": W_XGB, "w_rf": W_RF, "w_dc": W_DC},
            MODELS_DIR / "best_model.pkl")
joblib.dump(scaler,   MODELS_DIR / "scaler.pkl")
joblib.dump(le,       MODELS_DIR / "label_encoder.pkl")
joblib.dump(ENG_COLS, MODELS_DIR / "feature_cols.pkl")

meta = {
    "best_model": "XGB(cal)×4 + RF(cal)×1 + DC×5 — manual blend, no LR",
    "feature_cols": ENG_COLS,
    "engineered_features": [
        "elo_diff_sq", "home_form_momentum", "away_form_momentum",
        "home_goal_diff_form", "away_goal_diff_form", "net_goal_diff", "h2h_confidence"
    ],
    "classes": list(le.classes_),
    "metrics": {"accuracy": round(acc,4), "f1_macro": round(f1,4), "log_loss": round(ll,4)},
    "baseline_log_loss": 0.8333,
    "improvement": round(0.8333 - ll, 4),
}
with open(MODELS_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Models saved. Improvement over baseline: {meta['improvement']:.4f}")
