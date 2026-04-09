"""
EXP-11b: Squad features, training only on 2014+ matches where we have full data.
No NaN imputation needed — every match has squad features.
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import json
from scipy.stats import poisson

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Squad features
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

DIFF_FEATURES = [
    'squad_avg_overall_diff', 'squad_top3_avg_diff',
    'squad_value_diff', 'def_avg_diff', 'mid_avg_diff', 'fwd_avg_diff',
    'team_shooting_diff', 'team_passing_diff', 'team_defending_diff',
]

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


def join_squad_features(match_df, team_features):
    df = match_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = df['date'].dt.year

    available_years = sorted(team_features['year'].unique())
    tf_indexed = team_features.set_index(['team', 'year'])

    def get_fifa_year(match_year):
        candidates = [y for y in available_years if y <= match_year]
        return max(candidates) if candidates else None

    df['fifa_year'] = df['match_year'].apply(get_fifa_year)

    home_feats = []
    away_feats = []

    for _, row in df.iterrows():
        fy = row['fifa_year']
        home = row['home_team']
        away = row['away_team']
        hr, ar = {}, {}

        if fy is not None:
            if (home, fy) in tf_indexed.index:
                h = tf_indexed.loc[(home, fy)]
                for f in SQUAD_FEATURES:
                    hr[f'home_{f}'] = h.get(f, np.nan)
            else:
                for f in SQUAD_FEATURES:
                    hr[f'home_{f}'] = np.nan

            if (away, fy) in tf_indexed.index:
                a = tf_indexed.loc[(away, fy)]
                for f in SQUAD_FEATURES:
                    ar[f'away_{f}'] = a.get(f, np.nan)
            else:
                for f in SQUAD_FEATURES:
                    ar[f'away_{f}'] = np.nan
        else:
            for f in SQUAD_FEATURES:
                hr[f'home_{f}'] = np.nan
                ar[f'away_{f}'] = np.nan

        home_feats.append(hr)
        away_feats.append(ar)

    home_df = pd.DataFrame(home_feats, index=df.index)
    away_df = pd.DataFrame(away_feats, index=df.index)
    df = pd.concat([df, home_df, away_df], axis=1)

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
    print("EXP-11b: Squad Features (2014+ only)")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_dc.csv")
    team_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")

    # Filter train to 2014+
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_2014 = train_df[train_df['date'].dt.year >= 2014].copy()
    print(f"  Train (all): {len(train_df)}")
    print(f"  Train (2014+): {len(train_2014)}")
    print(f"  Test: {len(test_df)}")

    # Join squad features
    print("\nJoining squad features...")
    train_aug = join_squad_features(train_2014, team_features)
    test_aug = join_squad_features(test_df, team_features)

    # Drop rows where BOTH teams have no squad features
    home_squad_cols = [f'home_{f}' for f in SQUAD_FEATURES]
    away_squad_cols = [f'away_{f}' for f in SQUAD_FEATURES]

    train_has = train_aug['home_squad_avg_overall'].notna() & train_aug['away_squad_avg_overall'].notna()
    test_has = test_aug['home_squad_avg_overall'].notna() & test_aug['away_squad_avg_overall'].notna()

    train_clean = train_aug[train_has].copy()
    test_clean = test_aug[test_has].copy()

    # For test: also keep full test for baseline comparison
    print(f"  Train with full squad data: {len(train_clean)}/{len(train_aug)} ({len(train_clean)/len(train_aug)*100:.1f}%)")
    print(f"  Test with full squad data: {len(test_clean)}/{len(test_aug)} ({len(test_clean)/len(test_aug)*100:.1f}%)")

    # Feature sets
    ALL_FEATURES = BASE_FEATURES + home_squad_cols + away_squad_cols + DIFF_FEATURES

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_clean["outcome"].values)
    y_test_clean = le.transform(test_clean["outcome"].values)
    y_test_full = le.transform(test_aug["outcome"].values)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))

    # Model params
    XGB_P = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    RF_P = dict(n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=cw)
    LR_P = dict(C=1.0, max_iter=1000, class_weight=cw, solver="lbfgs", multi_class="multinomial")

    # DC model for blending
    dc_bundle = joblib.load(MODELS_DIR / "dc_model.pkl")
    attack_params = dc_bundle["attack"]
    defense_params = dc_bundle["defense"]
    team_idx_dc = dc_bundle["team_idx"]
    with open(MODELS_DIR / "dc_params.json") as f:
        dc_params = json.load(f)
    home_adv_dc = dc_params["home_adv"]
    rho_dc = dc_params["rho"]

    def dc_probs(home, away, neutral):
        if home not in team_idx_dc or away not in team_idx_dc:
            return np.array([0.30, 0.25, 0.45])
        goals = np.arange(11)
        hi, ai = team_idx_dc[home], team_idx_dc[away]
        ha = home_adv_dc if not neutral else 0.0
        lam = np.exp(attack_params[hi] + defense_params[ai] + ha)
        mu = np.exp(attack_params[ai] + defense_params[hi])
        sm = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        sm[0,0] *= max(1-lam*mu*rho_dc, 1e-10)
        sm[0,1] *= max(1+lam*rho_dc, 1e-10)
        sm[1,0] *= max(1+mu*rho_dc, 1e-10)
        sm[1,1] *= max(1-rho_dc, 1e-10)
        hw = float(np.sum(sm * (goals[:,None] > goals[None,:])))
        dr = float(np.sum(sm * (goals[:,None] == goals[None,:])))
        aw = 1.0-hw-dr
        t = hw+dr+aw
        return np.array([aw/t, dr/t, hw/t])

    results = []

    def evaluate(name, y_true, y_proba):
        ll = log_loss(y_true, y_proba)
        y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        results.append((name, ll, acc, f1, len(y_true)))
        print(f"  {name:55s}  ll={ll:.4f}  acc={acc:.4f}  f1={f1:.4f}  n={len(y_true)}")
        return ll

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # DC probs for test sets
    dc_test_clean = np.array([
        dc_probs(r['home_team'], r['away_team'], bool(r.get('neutral', r.get('neutral.1', True))))
        for _, r in test_clean.iterrows()
    ])
    dc_test_full = np.array([
        dc_probs(r['home_team'], r['away_team'], bool(r.get('neutral', r.get('neutral.1', True))))
        for _, r in test_aug.iterrows()
    ])

    # ── [A] Baseline: 59 features, ALL training data ──
    print("\n[A] Baseline: 59 features, full train (35K)")
    X_train_base_full = train_df[BASE_FEATURES].values
    X_test_base_full = test_df[BASE_FEATURES].values
    sc_a = StandardScaler()
    X_tr_a = sc_a.fit_transform(X_train_base_full)
    X_te_a = sc_a.transform(X_test_base_full)

    y_tr_a = le.transform(train_df["outcome"].values)
    ens_a = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
            ("lr", LogisticRegression(**LR_P, random_state=42)),
        ], voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_a.fit(X_tr_a, y_tr_a)
    p_a = ens_a.predict_proba(X_te_a)
    blend_a = (5*p_a + 1*dc_test_full) / 6
    evaluate("Baseline: 59 feat, 35K train, full test", y_test_full, blend_a)

    # ── [B] Baseline: 59 features, 2014+ training, clean test ──
    print("\n[B] Baseline: 59 features, 2014+ train")
    X_train_base = train_clean[BASE_FEATURES].values
    X_test_base = test_clean[BASE_FEATURES].values
    sc_b = StandardScaler()
    X_tr_b = sc_b.fit_transform(X_train_base)
    X_te_b = sc_b.transform(X_test_base)

    ens_b = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
            ("lr", LogisticRegression(**LR_P, random_state=42)),
        ], voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_b.fit(X_tr_b, y_train)
    p_b = ens_b.predict_proba(X_te_b)
    blend_b = (5*p_b + 1*dc_test_clean) / 6
    evaluate("Baseline: 59 feat, 2014+ train, clean test", y_test_clean, blend_b)

    # Also eval on full test (fill missing squad with 0 for non-covered teams)
    X_te_b_full = test_aug[BASE_FEATURES].values
    sc_b2 = StandardScaler()
    sc_b2.fit(train_clean[BASE_FEATURES].values)
    X_te_b_full_s = sc_b2.transform(X_te_b_full)
    p_b_full = ens_b.predict_proba(X_te_b_full_s)
    blend_b_full = (5*p_b_full + 1*dc_test_full) / 6
    evaluate("Baseline: 59 feat, 2014+ train, full test", y_test_full, blend_b_full)

    # ── [C] Squad features: 112 features, 2014+ train, clean test ──
    print("\n[C] + Squad features: 112 feat, 2014+ train")
    X_train_sq = train_clean[ALL_FEATURES].fillna(0).values
    X_test_sq = test_clean[ALL_FEATURES].fillna(0).values
    sc_c = StandardScaler()
    X_tr_c = sc_c.fit_transform(X_train_sq)
    X_te_c = sc_c.transform(X_test_sq)

    ens_c = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
            ("lr", LogisticRegression(**LR_P, random_state=42)),
        ], voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_c.fit(X_tr_c, y_train)
    p_c = ens_c.predict_proba(X_te_c)
    blend_c = (5*p_c + 1*dc_test_clean) / 6
    evaluate("Squad: 112 feat, 2014+ train, clean test", y_test_clean, blend_c)

    # ── [D] Only diff features: 68 feat ──
    print("\n[D] + Diff only: 68 feat, 2014+ train")
    DIFF_ONLY = BASE_FEATURES + DIFF_FEATURES
    X_train_d = train_clean[DIFF_ONLY].fillna(0).values
    X_test_d = test_clean[DIFF_ONLY].fillna(0).values
    sc_d = StandardScaler()
    X_tr_d = sc_d.fit_transform(X_train_d)
    X_te_d = sc_d.transform(X_test_d)

    ens_d = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
            ("lr", LogisticRegression(**LR_P, random_state=42)),
        ], voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_d.fit(X_tr_d, y_train)
    p_d = ens_d.predict_proba(X_te_d)
    blend_d = (5*p_d + 1*dc_test_clean) / 6
    evaluate("Diff only: 68 feat, 2014+ train, clean test", y_test_clean, blend_d)

    # ── [E] Core squad: 72 feat ──
    print("\n[E] + Core squad: 72 feat, 2014+ train")
    CORE = BASE_FEATURES + [
        'home_squad_avg_overall', 'away_squad_avg_overall',
        'home_squad_top3_avg', 'away_squad_top3_avg',
        'home_squad_total_value', 'away_squad_total_value',
        'home_def_avg', 'away_def_avg',
        'home_fwd_avg', 'away_fwd_avg',
        'squad_avg_overall_diff', 'squad_top3_avg_diff', 'squad_value_diff',
    ]
    X_train_e = train_clean[CORE].fillna(0).values
    X_test_e = test_clean[CORE].fillna(0).values
    sc_e = StandardScaler()
    X_tr_e = sc_e.fit_transform(X_train_e)
    X_te_e = sc_e.transform(X_test_e)

    ens_e = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
            ("lr", LogisticRegression(**LR_P, random_state=42)),
        ], voting="soft", weights=[4, 1, 0], n_jobs=-1)
    ens_e.fit(X_tr_e, y_train)
    p_e = ens_e.predict_proba(X_te_e)
    blend_e = (5*p_e + 1*dc_test_clean) / 6
    evaluate("Core squad: 72 feat, 2014+ train, clean test", y_test_clean, blend_e)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  {'Name':55s}  {'LL':>8s}  {'Acc':>7s}  {'F1':>7s}  {'N':>5s}")
    print("-" * 95)
    for name, ll, acc, f1, n in sorted(results, key=lambda x: x[1]):
        print(f"  {name:55s}  {ll:8.4f}  {acc:7.4f}  {f1:7.4f}  {n:5d}")

    # Feature importance for best squad model
    print("\n\nFeature Importance (model C: all squad features):")
    xgb_from = ens_c.named_estimators_['xgb']
    if hasattr(xgb_from, 'calibrated_classifiers_'):
        xgb_inner = xgb_from.calibrated_classifiers_[0].estimator
        if hasattr(xgb_inner, 'feature_importances_'):
            imp = xgb_inner.feature_importances_
            feat_imp = sorted(zip(ALL_FEATURES, imp), key=lambda x: -x[1])
            for i, (f, v) in enumerate(feat_imp[:35]):
                marker = " ★" if any(f.startswith(p) for p in ('home_squad','away_squad','squad_','home_def','away_def','home_mid','away_mid','home_fwd','away_fwd','home_gk','away_gk','home_team_','away_team_','home_strongest','away_strongest','home_weakest','away_weakest','def_avg','mid_avg','fwd_avg','team_')) else ""
                print(f"  {i+1:3d}. {f:50s}  {v:.4f}{marker}")

    # Save best squad model
    best_squad = min([r for r in results if 'Squad' in r[0] or 'Core' in r[0] or 'Diff' in r[0]], key=lambda x: x[1])
    best_base = min([r for r in results if 'Baseline' in r[0] and 'clean' in r[0]], key=lambda x: x[1])

    print(f"\n\nBest baseline (2014+ clean): {best_base[1]:.4f}")
    print(f"Best with squad features:    {best_squad[1]:.4f}")
    print(f"Delta:                       {best_squad[1] - best_base[1]:+.4f}")

    if best_squad[1] < best_base[1]:
        print("\n*** IMPROVEMENT! Saving model... ***")
        # Determine which
        if 'Core' in best_squad[0]:
            best_ens, best_sc, best_feats = ens_e, sc_e, CORE
        elif 'Diff' in best_squad[0]:
            best_ens, best_sc, best_feats = ens_d, sc_d, DIFF_ONLY
        else:
            best_ens, best_sc, best_feats = ens_c, sc_c, ALL_FEATURES

        bundle = {
            "xgb": best_ens.named_estimators_['xgb'],
            "rf": best_ens.named_estimators_['rf'],
            "w_xgb": 4, "w_rf": 1,
        }
        joblib.dump(bundle, MODELS_DIR / "best_model_exp11.pkl")
        joblib.dump(best_sc, MODELS_DIR / "scaler_exp11.pkl")
        joblib.dump(le, MODELS_DIR / "label_encoder_exp11.pkl")
        joblib.dump(best_feats, MODELS_DIR / "feature_cols_exp11.pkl")
        print("Saved!")


if __name__ == "__main__":
    main()
