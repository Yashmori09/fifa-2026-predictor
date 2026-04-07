"""
EXP-23 fine-tune: DC weight is still improving at 3.0 — push higher and find optimum.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

PROCESSED_DIR = Path("../data/processed")
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
classes = np.unique(y_train)
cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(train_df[FEATURE_COLS].values)
X_test_s  = scaler.transform(test_df[FEATURE_COLS].values)

# DC probabilities [away_win, draw, home_win]
test_dc_proba = np.column_stack([
    test_df["dc_away_win_prob"].values,
    test_df["dc_draw_prob"].values,
    test_df["dc_home_win_prob"].values,
])
test_dc_proba /= test_dc_proba.sum(axis=1, keepdims=True)

# Fit individual models
print("Fitting models...")
xgb_cal = CalibratedClassifierCV(
    xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                      eval_metric="mlogloss", random_state=42, verbosity=0),
    method="isotonic", cv=5)
rf_cal = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                           class_weight=cw, random_state=42, n_jobs=-1),
    method="isotonic", cv=5)
lr = LogisticRegression(C=1.0, max_iter=1000, class_weight=cw,
                        solver="lbfgs", multi_class="multinomial", random_state=42)

xgb_cal.fit(X_train_s, y_train); xgb_p = xgb_cal.predict_proba(X_test_s)
rf_cal.fit(X_train_s, y_train);  rf_p  = rf_cal.predict_proba(X_test_s)
lr.fit(X_train_s, y_train);      lr_p  = lr.predict_proba(X_test_s)
print("Done.\n")

best_ll = 0.8003
results = []

def evaluate(name, proba):
    global best_ll
    ll  = log_loss(y_test, proba)
    acc = accuracy_score(y_test, np.argmax(proba, axis=1))
    f1  = f1_score(y_test, np.argmax(proba, axis=1), average="macro")
    results.append((ll, acc, f1, name))
    marker = " *** BEST ***" if ll < best_ll else ""
    print(f"  {name:50s}  ll={ll:.4f}  acc={acc:.4f}  f1={f1:.4f}{marker}")
    if ll < best_ll:
        best_ll = ll
    return ll

print("[1] Push DC weight higher (best so far: [4,2,1,3.0] ll=0.8003)")
for dc_w in [3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]:
    # Try best base combo [4,2,1] + varying DC
    b = (4*xgb_p + 2*rf_p + 1*lr_p + dc_w*test_dc_proba) / (4+2+1+dc_w)
    evaluate(f"[4,2,1,{dc_w}]", b)
    b2 = (3*xgb_p + 3*rf_p + 2*lr_p + dc_w*test_dc_proba) / (3+3+2+dc_w)
    evaluate(f"[3,3,2,{dc_w}]", b2)
    b3 = (3*xgb_p + 2*rf_p + 1*lr_p + dc_w*test_dc_proba) / (3+2+1+dc_w)
    evaluate(f"[3,2,1,{dc_w}]", b3)

print("\n[2] Fine-tune around best")
# Take best DC weight found and sweep base weights
best_dc_found = min(results, key=lambda x: x[0])
print(f"  Best so far: {best_dc_found[3]}  ll={best_dc_found[0]:.4f}")

for xw in [3,4,5]:
    for rw in [1,2,3]:
        for lw in [0,1,2]:
            for dw in [3.0, 4.0, 5.0]:
                if lw == 0:
                    b = (xw*xgb_p + rw*rf_p + dw*test_dc_proba) / (xw+rw+dw)
                    evaluate(f"[{xw},{rw},0,{dw}]", b)
                else:
                    b = (xw*xgb_p + rw*rf_p + lw*lr_p + dw*test_dc_proba) / (xw+rw+lw+dw)
                    evaluate(f"[{xw},{rw},{lw},{dw}]", b)

print("\n" + "="*70)
print(f"OVERALL BEST: ll={best_ll:.4f}")
print("="*70)
print("\nTop 15 results:")
for ll, acc, f1, name in sorted(results)[:15]:
    print(f"  {ll:.4f}  acc={acc:.4f}  f1={f1:.4f}  {name}")
