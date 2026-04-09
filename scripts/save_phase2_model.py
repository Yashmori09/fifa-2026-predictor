"""
Save Phase 2 production model.
Config: XGB×3 + RF×1 ensemble, 97 features (EA + ELO minimal), best tuned XGB params.
Log loss: 0.8258 on test set.
"""
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── Load data ──
train_df = pd.read_csv(DATA_DIR / "train_dc.csv")
test_df = pd.read_csv(DATA_DIR / "test_dc.csv")
team_features = pd.read_csv(DATA_DIR / "team_features_by_year.csv")

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
train_df['year'] = train_df['date'].dt.year
test_df['year'] = test_df['date'].dt.year

# ── Squad features join + engineering (same as NB03) ──
SQUAD_FEATURES = [
    'squad_avg_overall', 'squad_median_overall', 'squad_std_overall',
    'squad_top3_avg', 'squad_bottom5_avg',
    'gk_avg', 'def_avg', 'mid_avg', 'fwd_avg',
    'strongest_unit', 'weakest_unit',
    'squad_total_value', 'squad_avg_value',
    'squad_avg_age', 'squad_avg_potential_gap', 'squad_avg_caps',
    'team_pace', 'team_shooting', 'team_passing',
    'team_dribbling', 'team_defending', 'team_physic',
]

def join_and_engineer(df):
    df = df.copy()
    avail = sorted(team_features['year'].unique())
    def get_yr(y):
        c = [x for x in avail if x <= y]
        return max(c) if c else None
    df['fifa_year'] = df['year'].apply(get_yr)

    htf = team_features.rename(columns={f: f'home_{f}' for f in SQUAD_FEATURES})
    htf = htf.rename(columns={'team': 'home_team', 'year': 'fifa_year'})
    atf = team_features.rename(columns={f: f'away_{f}' for f in SQUAD_FEATURES})
    atf = atf.rename(columns={'team': 'away_team', 'year': 'fifa_year'})

    home_cols = ['home_team', 'fifa_year'] + [f'home_{f}' for f in SQUAD_FEATURES]
    away_cols = ['away_team', 'fifa_year'] + [f'away_{f}' for f in SQUAD_FEATURES]

    df = df.merge(htf[home_cols], on=['home_team', 'fifa_year'], how='left')
    df = df.merge(atf[away_cols], on=['away_team', 'fifa_year'], how='left')

    for f in SQUAD_FEATURES:
        df[f'{f}_diff'] = df[f'home_{f}'] - df[f'away_{f}']

    df['overall_ratio'] = df['home_squad_avg_overall'] / df['away_squad_avg_overall'].clip(lower=1)
    df['top3_ratio'] = df['home_squad_top3_avg'] / df['away_squad_top3_avg'].clip(lower=1)
    df['value_ratio_log'] = np.log1p(df['home_squad_total_value']) - np.log1p(df['away_squad_total_value'])
    df['value_ratio'] = (df['home_squad_total_value'] + 1) / (df['away_squad_total_value'] + 1)

    df['home_squad_balance'] = df['home_strongest_unit'] - df['home_weakest_unit']
    df['away_squad_balance'] = df['away_strongest_unit'] - df['away_weakest_unit']
    df['squad_balance_diff'] = df['home_squad_balance'] - df['away_squad_balance']

    df['home_star_gap'] = df['home_squad_top3_avg'] - df['home_squad_avg_overall']
    df['away_star_gap'] = df['away_squad_top3_avg'] - df['away_squad_avg_overall']
    df['star_gap_diff'] = df['home_star_gap'] - df['away_star_gap']

    df['depth_diff'] = df['home_squad_bottom5_avg'] - df['away_squad_bottom5_avg']
    df['squad_std_diff'] = df['home_squad_std_overall'] - df['away_squad_std_overall']

    df['home_attack_vs_def'] = df['home_fwd_avg'] - df['away_def_avg']
    df['away_attack_vs_def'] = df['away_fwd_avg'] - df['home_def_avg']
    df['attack_vs_def_diff'] = df['home_attack_vs_def'] - df['away_attack_vs_def']
    df['mid_battle'] = df['home_mid_avg'] - df['away_mid_avg']
    df['gk_diff'] = df['home_gk_avg'] - df['away_gk_avg']

    df['pace_diff'] = df['home_team_pace'] - df['away_team_pace']
    df['physic_diff'] = df['home_team_physic'] - df['away_team_physic']
    df['shooting_diff'] = df['home_team_shooting'] - df['away_team_shooting']
    df['passing_diff'] = df['home_team_passing'] - df['away_team_passing']
    df['defending_diff'] = df['home_team_defending'] - df['away_team_defending']
    df['dribbling_diff'] = df['home_team_dribbling'] - df['away_team_dribbling']

    df['age_diff'] = df['home_squad_avg_age'] - df['away_squad_avg_age']
    df['caps_diff'] = df['home_squad_avg_caps'] - df['away_squad_avg_caps']
    df['potential_gap_diff'] = df['home_squad_avg_potential_gap'] - df['away_squad_avg_potential_gap']

    df['home_weighted_strength'] = 0.6 * df['home_squad_avg_overall'] + 0.4 * df['home_squad_top3_avg']
    df['away_weighted_strength'] = 0.6 * df['away_squad_avg_overall'] + 0.4 * df['away_squad_top3_avg']
    df['weighted_strength_diff'] = df['home_weighted_strength'] - df['away_weighted_strength']

    return df

print("Joining and engineering features...")
train_aug = join_and_engineer(train_df)
test_aug = join_and_engineer(test_df)

le = LabelEncoder()
y_train = le.fit_transform(train_aug['outcome'].values)
y_test = le.transform(test_aug['outcome'].values)

# ── Feature set: 97 features (NB02 best + elo_diff + elo_diff_sq + home/away_elo) ──
FORM = [
    'home_win_rate_5', 'home_avg_scored_5', 'home_avg_conceded_5',
    'home_pts_per_match_5', 'home_matches_played_5',
    'home_win_rate_10', 'home_avg_scored_10', 'home_avg_conceded_10',
    'home_pts_per_match_10', 'home_matches_played_10',
    'away_win_rate_5', 'away_avg_scored_5', 'away_avg_conceded_5',
    'away_pts_per_match_5', 'away_matches_played_5',
    'away_win_rate_10', 'away_avg_scored_10', 'away_avg_conceded_10',
    'away_pts_per_match_10', 'away_matches_played_10',
    'home_form_momentum', 'away_form_momentum',
    'home_goal_diff_form', 'away_goal_diff_form', 'net_goal_diff',
]

H2H = [
    'h2h_home_win_rate', 'h2h_home_avg_scored', 'h2h_home_avg_conceded',
    'h2h_total_meetings', 'h2h_recent_win_rate', 'h2h_confidence',
]

CONTEXT = [
    'neutral.1', 'tournament_importance',
    'home_conf_UEFA', 'home_conf_CAF', 'home_conf_AFC',
    'home_conf_CONCACAF', 'home_conf_CONMEBOL', 'home_conf_OFC', 'home_conf_UNKNOWN',
    'away_conf_UEFA', 'away_conf_CAF', 'away_conf_AFC',
    'away_conf_CONCACAF', 'away_conf_CONMEBOL', 'away_conf_OFC', 'away_conf_UNKNOWN',
    'same_confederation',
]

EA_ALL_DIFFS = [f'{f}_diff' for f in SQUAD_FEATURES]
EA_ENGINEERED = [
    'overall_ratio', 'top3_ratio', 'value_ratio_log', 'value_ratio',
    'squad_balance_diff', 'star_gap_diff', 'depth_diff', 'squad_std_diff',
    'home_attack_vs_def', 'away_attack_vs_def', 'attack_vs_def_diff',
    'mid_battle', 'gk_diff',
    'pace_diff', 'physic_diff', 'shooting_diff', 'passing_diff',
    'defending_diff', 'dribbling_diff',
    'age_diff', 'caps_diff', 'potential_gap_diff',
    'weighted_strength_diff',
]

FEATURE_COLS = (FORM + H2H + CONTEXT + EA_ALL_DIFFS + EA_ENGINEERED +
                ['elo_diff', 'elo_diff_sq', 'home_elo_before', 'away_elo_before'])

print(f"Feature count: {len(FEATURE_COLS)}")

# Verify all features exist
missing = [f for f in FEATURE_COLS if f not in train_aug.columns]
if missing:
    print(f"WARNING: missing features: {missing}")
    FEATURE_COLS = [f for f in FEATURE_COLS if f in train_aug.columns]

X_train = train_aug[FEATURE_COLS].values
X_test = test_aug[FEATURE_COLS].values

# ── Train XGB (tuned) ──
BEST_XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.5,
)

print("Training XGB (calibrated)...")
xgb_cal = CalibratedClassifierCV(
    xgb.XGBClassifier(**BEST_XGB_PARAMS, eval_metric='mlogloss', random_state=42, verbosity=0),
    method='isotonic', cv=5
)
xgb_cal.fit(X_train, y_train)
p_xgb = xgb_cal.predict_proba(X_test)
ll_xgb = log_loss(y_test, p_xgb)
print(f"  XGB log loss: {ll_xgb:.4f}")

# ── Train RF ──
RF_PARAMS = dict(n_estimators=500, max_depth=12, min_samples_leaf=5)
X_train_rf = np.nan_to_num(X_train, 0)
X_test_rf = np.nan_to_num(X_test, 0)

print("Training RF (calibrated)...")
rf_cal = CalibratedClassifierCV(
    RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1),
    method='isotonic', cv=5
)
rf_cal.fit(X_train_rf, y_train)
p_rf = rf_cal.predict_proba(X_test_rf)
ll_rf = log_loss(y_test, p_rf)
print(f"  RF log loss: {ll_rf:.4f}")

# ── Ensemble: XGB×3 + RF×1 ──
W_XGB, W_RF = 3, 1
p_blend = (W_XGB * p_xgb + W_RF * p_rf) / (W_XGB + W_RF)
ll_blend = log_loss(y_test, p_blend)
acc_blend = accuracy_score(y_test, np.argmax(p_blend, axis=1))
print(f"\n  Ensemble XGB×3 + RF×1: ll={ll_blend:.4f}, acc={acc_blend:.4f}")

# ── Save ──
bundle = {
    'xgb': xgb_cal,
    'rf': rf_cal,
    'w_xgb': W_XGB,
    'w_rf': W_RF,
    'feature_cols': FEATURE_COLS,
    'label_encoder': le,
    'xgb_params': BEST_XGB_PARAMS,
    'rf_params': RF_PARAMS,
    'phase': 2,
    'log_loss': ll_blend,
    'accuracy': acc_blend,
}

out_path = MODELS_DIR / "phase2_model.pkl"
joblib.dump(bundle, out_path)
print(f"\nSaved to {out_path}")
print(f"Phase 2 model: {len(FEATURE_COLS)} features, ll={ll_blend:.4f}, acc={acc_blend:.4f}")
