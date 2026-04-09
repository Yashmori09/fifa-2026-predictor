"""
EXP-11: Add squad quality features from EA FC ratings.

Hypothesis: Squad-level features (avg overall, positional strength, market value)
provide an independent signal that corrects confederation-inflated ELO/DC bias.

Approach:
- Join team_features_by_year.csv to existing train_dc/test_dc
- Add home_squad_*, away_squad_*, and diff features
- Pre-2014 matches → NaN (XGBoost handles missing values natively)
- Retrain same ensemble: XGB×4 + RF×1 + DC×1, isotonic calibration
- Compare log loss against baseline 0.8131
"""
import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Squad features to add (from team_features_by_year.csv) ──────
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

# Diff features we'll compute
DIFF_FEATURES = [
    'squad_avg_overall_diff', 'squad_top3_avg_diff',
    'squad_value_diff', 'def_avg_diff', 'mid_avg_diff', 'fwd_avg_diff',
    'team_shooting_diff', 'team_passing_diff', 'team_defending_diff',
]

# Original 59 features (must match exactly)
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


def join_squad_features(match_df: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join squad features to match data.
    Each match gets home_squad_* and away_squad_* features.
    Pre-2014 matches get NaN (no FIFA data available).
    """
    df = match_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = df['date'].dt.year

    # For each match year, find the closest FIFA year available
    # FIFA 15 = year 2014, ..., FC 26 = year 2025
    available_years = sorted(team_features['year'].unique())

    def get_fifa_year(match_year):
        """Map match year to closest available FIFA year."""
        if match_year < min(available_years):
            return None  # pre-2014, no data
        # Find the closest year <= match_year
        candidates = [y for y in available_years if y <= match_year]
        if candidates:
            return max(candidates)
        return min(available_years)

    df['fifa_year'] = df['match_year'].apply(get_fifa_year)

    # Create lookup: (team, year) -> features
    tf_indexed = team_features.set_index(['team', 'year'])

    # Add home squad features
    home_feats = []
    away_feats = []

    for _, row in df.iterrows():
        fy = row['fifa_year']
        home = row['home_team']
        away = row['away_team']

        home_row = {}
        away_row = {}

        if fy is not None:
            if (home, fy) in tf_indexed.index:
                hr = tf_indexed.loc[(home, fy)]
                for feat in SQUAD_FEATURES:
                    home_row[f'home_{feat}'] = hr.get(feat, np.nan)
            else:
                for feat in SQUAD_FEATURES:
                    home_row[f'home_{feat}'] = np.nan

            if (away, fy) in tf_indexed.index:
                ar = tf_indexed.loc[(away, fy)]
                for feat in SQUAD_FEATURES:
                    away_row[f'away_{feat}'] = ar.get(feat, np.nan)
            else:
                for feat in SQUAD_FEATURES:
                    away_row[f'away_{feat}'] = np.nan
        else:
            for feat in SQUAD_FEATURES:
                home_row[f'home_{feat}'] = np.nan
                away_row[f'away_{feat}'] = np.nan

        home_feats.append(home_row)
        away_feats.append(away_row)

    home_df = pd.DataFrame(home_feats, index=df.index)
    away_df = pd.DataFrame(away_feats, index=df.index)
    df = pd.concat([df, home_df, away_df], axis=1)

    # Compute diff features
    df['squad_avg_overall_diff'] = df['home_squad_avg_overall'] - df['away_squad_avg_overall']
    df['squad_top3_avg_diff'] = df['home_squad_top3_avg'] - df['away_squad_top3_avg']
    df['squad_value_diff'] = df['home_squad_total_value'] - df['away_squad_total_value']
    df['def_avg_diff'] = df['home_def_avg'] - df['away_def_avg']
    df['mid_avg_diff'] = df['home_mid_avg'] - df['away_mid_avg']
    df['fwd_avg_diff'] = df['home_fwd_avg'] - df['away_fwd_avg']
    df['team_shooting_diff'] = df['home_team_shooting'] - df['away_team_shooting']
    df['team_passing_diff'] = df['home_team_passing'] - df['away_team_passing']
    df['team_defending_diff'] = df['home_team_defending'] - df['away_team_defending']

    return df


def main():
    print("EXP-11: Squad Quality Features")
    print("=" * 70)

    # Load existing data
    print("\nLoading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_dc.csv")
    team_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")

    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Team features: {team_features.shape}")

    # Join squad features
    print("\nJoining squad features to training data...")
    train_aug = join_squad_features(train_df, team_features)
    test_aug = join_squad_features(test_df, team_features)

    # Check coverage
    train_has_squad = train_aug['home_squad_avg_overall'].notna().sum()
    test_has_squad = test_aug['home_squad_avg_overall'].notna().sum()
    print(f"  Train matches with squad features: {train_has_squad}/{len(train_aug)} ({train_has_squad/len(train_aug)*100:.1f}%)")
    print(f"  Test matches with squad features: {test_has_squad}/{len(test_aug)} ({test_has_squad/len(test_aug)*100:.1f}%)")

    # Build feature lists
    home_squad_cols = [f'home_{f}' for f in SQUAD_FEATURES]
    away_squad_cols = [f'away_{f}' for f in SQUAD_FEATURES]
    ALL_FEATURES = BASE_FEATURES + home_squad_cols + away_squad_cols + DIFF_FEATURES

    print(f"\n  Base features: {len(BASE_FEATURES)}")
    print(f"  Home squad features: {len(home_squad_cols)}")
    print(f"  Away squad features: {len(away_squad_cols)}")
    print(f"  Diff features: {len(DIFF_FEATURES)}")
    print(f"  Total features: {len(ALL_FEATURES)}")

    # Prepare X, y
    le = LabelEncoder()
    y_train = le.fit_transform(train_aug["outcome"].values)
    y_test = le.transform(test_aug["outcome"].values)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))

    X_train = train_aug[ALL_FEATURES].values.astype(float)
    X_test = test_aug[ALL_FEATURES].values.astype(float)

    # Scale (StandardScaler handles NaN by passing through for XGBoost)
    # But sklearn StandardScaler doesn't handle NaN — we need to be careful
    # Option: impute NaN with 0 for scaler, but XGBoost can handle NaN natively
    # So: don't scale squad features, or use a NaN-safe approach

    # Approach: scale base features, leave squad features as-is for XGBoost
    # Actually, let's just impute NaN with 0 for scaling (XGBoost doesn't need scaling,
    # but RF and LR do). The NaN indicator is captured by the fact that ALL squad
    # features are 0 simultaneously for pre-2014 matches.
    X_train_filled = np.nan_to_num(X_train, nan=0.0)
    X_test_filled = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_filled)
    X_test_s = scaler.transform(X_test_filled)

    # Also prepare XGBoost-specific versions with NaN preserved
    # XGBoost handles NaN natively — it learns optimal split direction for missing values
    X_train_xgb = X_train.copy()  # keeps NaN
    X_test_xgb = X_test.copy()

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    results = []

    def evaluate(name, y_pred_proba):
        ll = log_loss(y_test, y_pred_proba)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append((name, ll, acc, f1))
        print(f"  {name:50s}  ll={ll:.4f}  acc={acc:.4f}  f1={f1:.4f}")
        return ll

    # ── Baseline (original 59 features, same hyperparams) ──────
    print("\n[0] Baseline reproduction (59 features)")
    X_train_base = train_aug[BASE_FEATURES].values
    X_test_base = test_aug[BASE_FEATURES].values
    scaler_base = StandardScaler()
    X_train_base_s = scaler_base.fit_transform(X_train_base)
    X_test_base_s = scaler_base.transform(X_test_base)

    XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=cw)
    LR_PARAMS = dict(C=1.0, max_iter=1000, class_weight=cw, solver="lbfgs", multi_class="multinomial")

    xgb_base = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf_base = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    lr_base = LogisticRegression(**LR_PARAMS, random_state=42)

    ens_base = VotingClassifier(
        estimators=[("xgb", xgb_base), ("rf", rf_base), ("lr", lr_base)],
        voting="soft", weights=[4, 1, 0], n_jobs=-1)

    # DC voter weights: for log loss eval, use DC×5; for simulation, DC×1
    # In the ensemble, LR weight=0 effectively removes it
    # The actual blend in predictor.py is: (XGB×4 + RF×1 + DC×1) / 6
    # But in training, we use VotingClassifier with XGB×4+RF×1+LR×0 (no DC in sklearn)
    # DC is added post-hoc during prediction

    ens_base.fit(X_train_base_s, y_train)
    base_proba = ens_base.predict_proba(X_test_base_s)

    # Add DC voter post-hoc (like predictor.py does)
    # Load DC model for test set predictions
    import json
    dc_bundle = joblib.load(MODELS_DIR / "dc_model.pkl")
    attack_params = dc_bundle["attack"]
    defense_params = dc_bundle["defense"]
    team_idx_dc = dc_bundle["team_idx"]
    with open(MODELS_DIR / "dc_params.json") as f:
        dc_params = json.load(f)
    home_adv_dc = dc_params["home_adv"]
    rho_dc = dc_params["rho"]

    from scipy.stats import poisson
    MAX_GOALS = 10

    def dc_probs(home, away, neutral):
        if home not in team_idx_dc or away not in team_idx_dc:
            return np.array([0.30, 0.25, 0.45])  # [away, draw, home]
        goals = np.arange(MAX_GOALS + 1)
        hi, ai = team_idx_dc[home], team_idx_dc[away]
        ha = home_adv_dc if not neutral else 0.0
        lam = np.exp(attack_params[hi] + defense_params[ai] + ha)
        mu = np.exp(attack_params[ai] + defense_params[hi])
        sm = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        sm[0,0] *= max(1 - lam*mu*rho_dc, 1e-10)
        sm[0,1] *= max(1 + lam*rho_dc, 1e-10)
        sm[1,0] *= max(1 + mu*rho_dc, 1e-10)
        sm[1,1] *= max(1 - rho_dc, 1e-10)
        hw = np.sum(sm[np.arange(11)[:,None] > np.arange(11)[None,:]] * sm[np.arange(11)[:,None] > np.arange(11)[None,:]])
        # Simpler:
        home_mask = goals[:, None] > goals[None, :]
        draw_mask = goals[:, None] == goals[None, :]
        hw = float(np.sum(sm * home_mask))
        dr = float(np.sum(sm * draw_mask))
        aw = 1.0 - hw - dr
        t = hw + dr + aw
        # Order: [away_win, draw, home_win] to match le.classes_
        return np.array([aw/t, dr/t, hw/t])

    # Compute DC probs for test set
    dc_test_probs = np.array([
        dc_probs(row['home_team'], row['away_team'],
                 bool(row.get('neutral', row.get('neutral.1', True))))
        for _, row in test_df.iterrows()
    ])

    # Blend: (XGB×4 + RF×1) from VotingClassifier + DC×1
    # VotingClassifier already weights XGB×4 + RF×1 internally (weights=[4,1,0])
    # So base_proba = (4*xgb + 1*rf) / 5
    # Final blend: (5*base_proba + 1*dc) / 6 = (4*xgb + 1*rf + 1*dc) / 6
    W_DC = 1
    blended_base = (5 * base_proba + W_DC * dc_test_probs) / (5 + W_DC)
    evaluate("Baseline (59 features, XGB×4+RF×1+DC×1)", blended_base)

    # ── EXP-11a: All squad features (59 + 22×2 + 9 diff = 112) ──────
    print("\n[1] EXP-11a: + all squad features (112 total)")

    xgb_squad = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf_squad = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    lr_squad = LogisticRegression(**LR_PARAMS, random_state=42)

    ens_squad = VotingClassifier(
        estimators=[("xgb", xgb_squad), ("rf", rf_squad), ("lr", lr_squad)],
        voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_squad.fit(X_train_s, y_train)
    squad_proba = ens_squad.predict_proba(X_test_s)
    blended_squad = (5 * squad_proba + W_DC * dc_test_probs) / (5 + W_DC)
    ll_all = evaluate("EXP-11a: + all 53 squad features", blended_squad)

    # ── EXP-11b: Only diff features (59 + 9 = 68) ──────
    print("\n[2] EXP-11b: + only diff features (68 total)")
    DIFF_ONLY_FEATURES = BASE_FEATURES + DIFF_FEATURES
    X_train_diff = np.nan_to_num(train_aug[DIFF_ONLY_FEATURES].values.astype(float), nan=0.0)
    X_test_diff = np.nan_to_num(test_aug[DIFF_ONLY_FEATURES].values.astype(float), nan=0.0)
    scaler_diff = StandardScaler()
    X_train_diff_s = scaler_diff.fit_transform(X_train_diff)
    X_test_diff_s = scaler_diff.transform(X_test_diff)

    xgb_diff = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf_diff = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    lr_diff = LogisticRegression(**LR_PARAMS, random_state=42)

    ens_diff = VotingClassifier(
        estimators=[("xgb", xgb_diff), ("rf", rf_diff), ("lr", lr_diff)],
        voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_diff.fit(X_train_diff_s, y_train)
    diff_proba = ens_diff.predict_proba(X_test_diff_s)
    blended_diff = (5 * diff_proba + W_DC * dc_test_probs) / (5 + W_DC)
    ll_diff = evaluate("EXP-11b: + only 9 diff features", blended_diff)

    # ── EXP-11c: Core squad features only (59 + selected) ──────
    print("\n[3] EXP-11c: + core squad features only")
    CORE_SQUAD = [
        'home_squad_avg_overall', 'away_squad_avg_overall',
        'home_squad_top3_avg', 'away_squad_top3_avg',
        'home_squad_total_value', 'away_squad_total_value',
        'home_def_avg', 'away_def_avg',
        'home_fwd_avg', 'away_fwd_avg',
        'squad_avg_overall_diff', 'squad_top3_avg_diff', 'squad_value_diff',
    ]
    CORE_FEATURES = BASE_FEATURES + CORE_SQUAD
    X_train_core = np.nan_to_num(train_aug[CORE_FEATURES].values.astype(float), nan=0.0)
    X_test_core = np.nan_to_num(test_aug[CORE_FEATURES].values.astype(float), nan=0.0)
    scaler_core = StandardScaler()
    X_train_core_s = scaler_core.fit_transform(X_train_core)
    X_test_core_s = scaler_core.transform(X_test_core)

    xgb_core = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf_core = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    lr_core = LogisticRegression(**LR_PARAMS, random_state=42)

    ens_core = VotingClassifier(
        estimators=[("xgb", xgb_core), ("rf", rf_core), ("lr", lr_core)],
        voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_core.fit(X_train_core_s, y_train)
    core_proba = ens_core.predict_proba(X_test_core_s)
    blended_core = (5 * core_proba + W_DC * dc_test_probs) / (5 + W_DC)
    ll_core = evaluate("EXP-11c: + 13 core squad features", blended_core)

    # ── Summary ──────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Name':50s}  {'LL':>8s}  {'Acc':>7s}  {'F1':>7s}  {'ΔLL':>8s}")
    print("-" * 85)
    baseline_ll = results[0][1]
    for name, ll, acc, f1 in sorted(results, key=lambda x: x[1]):
        delta = ll - baseline_ll
        marker = " ← BEST" if ll == min(r[1] for r in results) else ""
        print(f"  {name:50s}  {ll:8.4f}  {acc:7.4f}  {f1:7.4f}  {delta:+8.4f}{marker}")

    # ── Feature importance for best model ──────
    print("\n\nFeature Importance (XGBoost, all-features model):")
    # Get XGBoost from the ensemble
    xgb_from_ens = ens_squad.named_estimators_['xgb']
    # CalibratedClassifierCV wraps the actual model
    if hasattr(xgb_from_ens, 'calibrated_classifiers_'):
        xgb_inner = xgb_from_ens.calibrated_classifiers_[0].estimator
        if hasattr(xgb_inner, 'feature_importances_'):
            importances = xgb_inner.feature_importances_
            feat_imp = sorted(zip(ALL_FEATURES, importances), key=lambda x: -x[1])
            for i, (feat, imp) in enumerate(feat_imp[:30]):
                marker = " ★" if feat.startswith(('home_squad', 'away_squad', 'squad_', 'def_', 'mid_', 'fwd_', 'team_')) else ""
                print(f"  {i+1:3d}. {feat:45s}  {imp:.4f}{marker}")

    # Save best model if improved
    best_result = min(results, key=lambda x: x[1])
    if best_result[1] < baseline_ll:
        print(f"\n{'='*70}")
        print(f"IMPROVEMENT: {best_result[0]}")
        print(f"  Log loss: {baseline_ll:.4f} → {best_result[1]:.4f} (Δ = {best_result[1]-baseline_ll:+.4f})")
        print(f"\n  Saving new model...")

        # Determine which model was best and save it
        if best_result[0].endswith("squad features"):
            best_ens = ens_squad
            best_scaler = scaler
            best_features = ALL_FEATURES
        elif best_result[0].endswith("diff features"):
            best_ens = ens_diff
            best_scaler = scaler_diff
            best_features = DIFF_ONLY_FEATURES
        elif best_result[0].endswith("squad features"):
            best_ens = ens_core
            best_scaler = scaler_core
            best_features = CORE_FEATURES

        # Save
        bundle = {
            "xgb": best_ens.named_estimators_['xgb'],
            "rf": best_ens.named_estimators_['rf'],
            "w_xgb": 4,
            "w_rf": 1,
        }
        joblib.dump(bundle, MODELS_DIR / "best_model_exp11.pkl")
        joblib.dump(best_scaler, MODELS_DIR / "scaler_exp11.pkl")
        joblib.dump(le, MODELS_DIR / "label_encoder_exp11.pkl")
        joblib.dump(best_features, MODELS_DIR / "feature_cols_exp11.pkl")
        print(f"  Saved: best_model_exp11.pkl, scaler_exp11.pkl, feature_cols_exp11.pkl")
    else:
        print(f"\nNo improvement over baseline ({baseline_ll:.4f})")


if __name__ == "__main__":
    main()
