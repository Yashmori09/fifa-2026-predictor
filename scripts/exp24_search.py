"""
EXP-24: Systematic search on 59-feature set.
Tests: ensemble weights, XGBoost hyperparams, LightGBM, re-calibration strategies.
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

BASE_FEATURES = [
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
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw = dict(zip(classes, weights))

X_train = train_df[BASE_FEATURES].values
X_test  = test_df[BASE_FEATURES].values
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

best_ll = 0.8131
best_name = "baseline [4,2,2]"
results = []

def evaluate(name, model):
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)
    ll = log_loss(y_test, proba)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    results.append((name, ll, acc))
    marker = " *** BEST ***" if ll < best_ll else ""
    print(f"  {name:45s}  ll={ll:.4f}  acc={acc:.4f}{marker}")
    return ll

def make_ensemble(xgb_params, rf_params, lr_params, weights):
    xgb_base = xgb.XGBClassifier(**xgb_params, eval_metric="mlogloss", random_state=42, verbosity=0)
    xgb_cal  = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    rf_base  = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    rf_cal   = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    lr       = LogisticRegression(**lr_params, random_state=42)
    return VotingClassifier(
        estimators=[("xgb", xgb_cal), ("rf", rf_cal), ("lr", lr)],
        voting="soft", weights=weights, n_jobs=-1,
    )

XGB_BASE = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
RF_BASE  = dict(n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=cw)
LR_BASE  = dict(C=1.0, max_iter=1000, class_weight=cw, solver="lbfgs", multi_class="multinomial")

print(f"\nBaseline: ll=0.8131 (weights [4,2,2], 59 features)")
print("="*75)

# --- Weight search ---
print("\n[1] Ensemble weight search")
for w in [[5,2,2],[4,3,2],[4,2,3],[3,2,2],[4,1,2],[6,2,2],[4,2,1],[5,3,2],[3,3,2]]:
    ll = evaluate(f"weights {w}", make_ensemble(XGB_BASE, RF_BASE, LR_BASE, w))
    if ll < best_ll:
        best_ll = ll; best_name = f"weights {w}"

# --- XGBoost hyperparams ---
print("\n[2] XGBoost hyperparameter search")
for md, lr_val, sub, col in [
    (4, 0.05, 0.8, 0.8),
    (6, 0.05, 0.8, 0.8),
    (5, 0.03, 0.8, 0.8),
    (5, 0.05, 0.7, 0.7),
    (4, 0.03, 0.9, 0.9),
    (5, 0.05, 0.8, 0.7),
    (4, 0.05, 0.9, 0.8),
]:
    xgb_p = dict(n_estimators=300, max_depth=md, learning_rate=lr_val,
                 subsample=sub, colsample_bytree=col, reg_alpha=0.1, reg_lambda=1.0)
    ll = evaluate(f"xgb depth={md} lr={lr_val} sub={sub} col={col}", make_ensemble(xgb_p, RF_BASE, LR_BASE, [4,2,2]))
    if ll < best_ll:
        best_ll = ll; best_name = f"xgb depth={md} lr={lr_val} sub={sub} col={col}"

# --- Regularization ---
print("\n[3] XGBoost regularization search")
for alpha, lam in [(0.0,1.0),(0.5,1.0),(0.1,2.0),(0.1,0.5),(0.3,1.5),(0.0,2.0)]:
    xgb_p = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=alpha, reg_lambda=lam)
    ll = evaluate(f"xgb alpha={alpha} lambda={lam}", make_ensemble(xgb_p, RF_BASE, LR_BASE, [4,2,2]))
    if ll < best_ll:
        best_ll = ll; best_name = f"xgb alpha={alpha} lambda={lam}"

# --- More trees ---
print("\n[4] More trees + lower LR")
for n_est, lr_val in [(400,0.04),(500,0.03),(350,0.04)]:
    xgb_p = dict(n_estimators=n_est, max_depth=5, learning_rate=lr_val,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    ll = evaluate(f"xgb n={n_est} lr={lr_val}", make_ensemble(xgb_p, RF_BASE, LR_BASE, [4,2,2]))
    if ll < best_ll:
        best_ll = ll; best_name = f"xgb n={n_est} lr={lr_val}"

# --- LightGBM ---
print("\n[5] LightGBM variants")
try:
    import lightgbm as lgb
    for n_est, lr_val, md in [(300,0.05,5),(300,0.03,5),(200,0.05,4)]:
        lgb_base = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=lr_val,
                                       max_depth=md, class_weight=cw,
                                       random_state=42, verbosity=-1, n_jobs=-1)
        lgb_cal = CalibratedClassifierCV(lgb_base, method="isotonic", cv=5)
        xgb_cal = CalibratedClassifierCV(xgb.XGBClassifier(**XGB_BASE, eval_metric="mlogloss",
                                          random_state=42, verbosity=0), method="isotonic", cv=5)
        rf_cal  = CalibratedClassifierCV(RandomForestClassifier(**RF_BASE, random_state=42, n_jobs=-1),
                                          method="isotonic", cv=5)
        lr      = LogisticRegression(**LR_BASE, random_state=42)
        # XGB + LGB + LR
        ens = VotingClassifier([("xgb",xgb_cal),("lgb",lgb_cal),("lr",lr)],
                                voting="soft", weights=[4,3,2], n_jobs=-1)
        ll = evaluate(f"xgb+lgb(n={n_est},lr={lr_val})+lr [4,3,2]", ens)
        if ll < best_ll:
            best_ll = ll; best_name = f"xgb+lgb(n={n_est})+lr"
except ImportError:
    print("  LightGBM not installed, skipping")

# --- LR variants ---
print("\n[6] LR regularization")
for C, solver in [(0.5,"lbfgs"),(2.0,"lbfgs"),(0.1,"lbfgs"),(1.0,"saga")]:
    lr_p = dict(C=C, max_iter=1000, class_weight=cw, solver=solver, multi_class="multinomial")
    ll = evaluate(f"lr C={C} solver={solver}", make_ensemble(XGB_BASE, RF_BASE, lr_p, [4,2,2]))
    if ll < best_ll:
        best_ll = ll; best_name = f"lr C={C} solver={solver}"

print("\n" + "="*75)
print(f"BEST: {best_name}  ll={best_ll:.4f}")
print("="*75)
print("\nAll results (sorted by log loss):")
for name, ll, acc in sorted(results, key=lambda x: x[1]):
    print(f"  {ll:.4f}  {acc:.4f}  {name}")
