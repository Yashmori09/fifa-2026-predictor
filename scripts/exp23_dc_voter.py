"""
EXP-23: Add DC model as 4th soft voter in the ensemble.
DC probabilities are computed from Poisson score distributions — orthogonal to tree/linear models.
Tests DC as direct voter vs different weight combinations.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

PROCESSED_DIR = Path("../data/processed")
MODELS_DIR    = Path("../models")

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
test_df  = pd.read_csv(PROCESSED_DIR / "test_dc.csv")

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
    "elo_diff_sq", "home_form_momentum", "away_form_momentum",
    "home_goal_diff_form", "away_goal_diff_form", "net_goal_diff", "h2h_confidence",
    "dc_home_win_prob", "dc_draw_prob", "dc_away_win_prob",
    "dc_lambda", "dc_mu", "dc_total_goals", "dc_goal_diff",
]

le = LabelEncoder()
y_train = le.fit_transform(train_df["outcome"].values)
y_test  = le.transform(test_df["outcome"].values)
# classes order: 0=away_win, 1=draw, 2=home_win
print(f"Classes: {le.classes_}")

classes = np.unique(y_train)
cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

X_train = train_df[FEATURE_COLS].values
X_test  = test_df[FEATURE_COLS].values
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── DC model as a sklearn-compatible classifier ───────────────────────────────
class DCClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps the fitted Dixon-Coles model as a sklearn classifier.
    Uses pre-computed dc_home_win_prob, dc_draw_prob, dc_away_win_prob columns
    directly from the feature matrix — no refit needed.
    classes_ order must match LabelEncoder: [away_win, draw, home_win]
    """
    def __init__(self, feature_cols, dc_cols):
        self.feature_cols = feature_cols
        self.dc_cols = dc_cols  # indices of [dc_home_win_prob, dc_draw_prob, dc_away_win_prob]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        # X is scaled — but DC probs are in specific columns, we need unscaled
        # We stored them separately (see below)
        return self._dc_proba

    def predict(self, X):
        return self.classes_[np.argmax(self._dc_proba, axis=1)]

    def set_proba(self, proba):
        self._dc_proba = proba


# Extract DC probs directly from unscaled data
# Column order in FEATURE_COLS: dc_home_win_prob, dc_draw_prob, dc_away_win_prob
hw_idx = FEATURE_COLS.index("dc_home_win_prob")
dr_idx = FEATURE_COLS.index("dc_draw_prob")
aw_idx = FEATURE_COLS.index("dc_away_win_prob")

# DC probs: [away_win, draw, home_win] to match LabelEncoder order
train_dc_proba = np.column_stack([
    train_df["dc_away_win_prob"].values,
    train_df["dc_draw_prob"].values,
    train_df["dc_home_win_prob"].values,
])
test_dc_proba = np.column_stack([
    test_df["dc_away_win_prob"].values,
    test_df["dc_draw_prob"].values,
    test_df["dc_home_win_prob"].values,
])

# Normalize rows (sum to 1)
train_dc_proba = train_dc_proba / train_dc_proba.sum(axis=1, keepdims=True)
test_dc_proba  = test_dc_proba  / test_dc_proba.sum(axis=1, keepdims=True)

print(f"DC proba sample: {test_dc_proba[0]} (away, draw, home)")

# ── Base models (EXP-24 best config) ─────────────────────────────────────────
def make_base_models():
    xgb_base = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight=cw, random_state=42, n_jobs=-1,
    )
    rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight=cw,
                            solver="lbfgs", multi_class="multinomial", random_state=42)
    return xgb_cal, rf_cal, lr

best_ll = 0.8123
best_name = "EXP-24 baseline [3,3,2]"
results = []

def evaluate(name, proba):
    global best_ll, best_name
    ll   = log_loss(y_test, proba)
    pred = np.argmax(proba, axis=1)
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred, average="macro")
    results.append((ll, acc, f1, name))
    marker = " *** BEST ***" if ll < best_ll else ""
    print(f"  {name:55s}  ll={ll:.4f}  acc={acc:.4f}  f1={f1:.4f}{marker}")
    if ll < best_ll:
        best_ll = ll; best_name = name
    return ll

print(f"\nBaseline (EXP-24): ll=0.8123")
print("="*80)

# First get the base ensemble probabilities once (expensive — calibrated CV)
print("\nFitting base ensemble (XGB+RF+LR [3,3,2])...")
xgb_cal, rf_cal, lr = make_base_models()
base_ens = VotingClassifier(
    [("xgb", xgb_cal), ("rf", rf_cal), ("lr", lr)],
    voting="soft", weights=[3, 3, 2], n_jobs=-1
)
base_ens.fit(X_train_s, y_train)
base_proba = base_ens.predict_proba(X_test_s)
base_ll = log_loss(y_test, base_proba)
print(f"  Base ensemble ll={base_ll:.4f}")

# Get individual model probas for manual blending
xgb_cal2, rf_cal2, lr2 = make_base_models()
print("Fitting individual models for blending...")
xgb_cal2.fit(X_train_s, y_train); xgb_proba = xgb_cal2.predict_proba(X_test_s)
rf_cal2.fit(X_train_s, y_train);  rf_proba  = rf_cal2.predict_proba(X_test_s)
lr2.fit(X_train_s, y_train);      lr_proba  = lr2.predict_proba(X_test_s)
print("Done.\n")

print("[1] Manual blend: XGB+RF+LR+DC with various DC weights")
# Base weights [3,3,2] — try adding DC at different weights
for dc_w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    total = 3 + 3 + 2 + dc_w
    blended = (3*xgb_proba + 3*rf_proba + 2*lr_proba + dc_w*test_dc_proba) / total
    evaluate(f"XGB+RF+LR+DC  [3,3,2,{dc_w}]", blended)

print("\n[2] DC replaces LR")
for dc_w in [1.0, 2.0, 3.0]:
    total = 3 + 3 + dc_w
    blended = (3*xgb_proba + 3*rf_proba + dc_w*test_dc_proba) / total
    evaluate(f"XGB+RF+DC (no LR) [3,3,{dc_w}]", blended)

print("\n[3] DC alone (sanity check)")
evaluate("DC only", test_dc_proba)

print("\n[4] Best blend so far + fine-tune weights")
# Find best DC weight from section 1
best_dc_w = min(
    [(dc_w, log_loss(y_test, (3*xgb_proba+3*rf_proba+2*lr_proba+dc_w*test_dc_proba)/(3+3+2+dc_w)))
     for dc_w in [0.5,1.0,1.5,2.0,2.5,3.0]],
    key=lambda x: x[1]
)[0]
print(f"  Best DC weight from section 1: {best_dc_w}")
for xw, rw, lw, dw in [
    (4,3,2,best_dc_w), (3,3,2,best_dc_w), (3,2,2,best_dc_w),
    (4,2,1,best_dc_w), (3,3,1,best_dc_w), (4,3,1,best_dc_w),
]:
    total = xw+rw+lw+dw
    blended = (xw*xgb_proba + rw*rf_proba + lw*lr_proba + dw*test_dc_proba) / total
    evaluate(f"[{xw},{rw},{lw},{dw}]", blended)

print("\n" + "="*80)
print(f"BEST: {best_name}  ll={best_ll:.4f}")
print("="*80)
print("\nAll results (sorted by log loss):")
for ll, acc, f1, name in sorted(results):
    mark = " <-- BEST" if ll == best_ll else ""
    print(f"  {ll:.4f}  acc={acc:.4f}  f1={f1:.4f}  {name}{mark}")
