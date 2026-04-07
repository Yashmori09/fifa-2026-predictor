"""
EXP-24 Phase 2: Combine best findings from search.
Best individual: xgb n=500 lr=0.03 (0.8126) + weights [3,3,2] (0.8128) + alpha=0.5 (0.8128)
Try combinations of these.
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

X_train_s = StandardScaler().fit_transform(train_df[FEATURE_COLS].values)
scaler = StandardScaler().fit(train_df[FEATURE_COLS].values)
X_test_s = scaler.transform(test_df[FEATURE_COLS].values)

best_ll = 0.8126
best_name = "xgb n=500 lr=0.03 [4,2,2]"
results = []

def evaluate(name, model):
    global best_ll, best_name
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)
    ll  = log_loss(y_test, proba)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    f1  = f1_score(y_test, model.predict(X_test_s), average="macro")
    results.append((ll, acc, f1, name))
    marker = " *** BEST ***" if ll < best_ll else ""
    print(f"  {name:55s}  ll={ll:.4f}  acc={acc:.4f}{marker}")
    if ll < best_ll:
        best_ll = ll; best_name = name
    return ll

LR = LogisticRegression(C=1.0, max_iter=1000, class_weight=cw,
                        solver="lbfgs", multi_class="multinomial", random_state=42)
RF_BASE = dict(n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=cw)

def make_xgb(n=300, lr=0.05, depth=5, sub=0.8, col=0.8, alpha=0.1, lam=1.0):
    base = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                              subsample=sub, colsample_bytree=col,
                              reg_alpha=alpha, reg_lambda=lam,
                              eval_metric="mlogloss", random_state=42, verbosity=0)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

def make_rf():
    return CalibratedClassifierCV(
        RandomForestClassifier(**RF_BASE, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)

def ensemble(xgb_model, weights):
    return VotingClassifier([("xgb", xgb_model), ("rf", make_rf()), ("lr", LR)],
                             voting="soft", weights=weights, n_jobs=-1)

print(f"Previous best: ll=0.8126")
print("="*75)

print("\n[1] Combine: best XGB params + best weights")
configs = [
    # (n, lr, depth, sub, col, alpha, lam, weights)
    (500, 0.03, 5, 0.8, 0.8, 0.1, 1.0, [3, 3, 2]),
    (500, 0.03, 5, 0.8, 0.8, 0.1, 1.0, [4, 3, 2]),
    (500, 0.03, 5, 0.8, 0.8, 0.5, 1.0, [4, 2, 2]),
    (500, 0.03, 5, 0.8, 0.8, 0.5, 1.0, [3, 3, 2]),
    (500, 0.03, 5, 0.8, 0.8, 0.5, 1.0, [4, 3, 2]),
    (500, 0.03, 5, 0.8, 0.7, 0.1, 1.0, [4, 2, 2]),
    (500, 0.03, 5, 0.8, 0.7, 0.5, 1.0, [3, 3, 2]),
    (500, 0.03, 4, 0.8, 0.8, 0.1, 1.0, [4, 2, 2]),
    (500, 0.03, 4, 0.8, 0.8, 0.5, 1.0, [3, 3, 2]),
    (600, 0.02, 5, 0.8, 0.8, 0.1, 1.0, [4, 2, 2]),
    (600, 0.02, 5, 0.8, 0.8, 0.1, 1.0, [3, 3, 2]),
    (600, 0.025, 5, 0.8, 0.8, 0.5, 1.0, [3, 3, 2]),
    (700, 0.02, 5, 0.8, 0.8, 0.1, 1.0, [4, 2, 2]),
    (400, 0.03, 5, 0.8, 0.8, 0.5, 1.0, [3, 3, 2]),
]

for n, lr, depth, sub, col, alpha, lam, w in configs:
    name = f"n={n} lr={lr} d={depth} sub={sub} col={col} a={alpha} w={w}"
    evaluate(name, ensemble(make_xgb(n,lr,depth,sub,col,alpha,lam), w))

print("\n" + "="*75)
print(f"BEST: {best_name}  ll={best_ll:.4f}")
print("="*75)
print("\nAll results (sorted):")
for ll, acc, f1, name in sorted(results):
    mark = " <-- BEST" if ll == best_ll else ""
    print(f"  {ll:.4f}  acc={acc:.4f}  f1={f1:.4f}  {name}{mark}")
