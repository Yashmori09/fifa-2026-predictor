"""
EXP-11c: Retrain with enriched squad data (EA + FM23 + imputation).
Coverage: 6.3% -> 95.5% of training data now has squad features.

Compares:
  A) EA-only features (on squad-covered subset)
  B) Baseline 59 features (no EA)
  C) Combined (base + EA) with XGBoost NaN handling
"""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib, json
from scipy.stats import poisson

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

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
    'squad_avg_overall_diff', 'squad_top3_avg_diff', 'squad_value_diff',
    'def_avg_diff', 'mid_avg_diff', 'fwd_avg_diff',
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

HOME_SQUAD = [f'home_{f}' for f in SQUAD_FEATURES]
AWAY_SQUAD = [f'away_{f}' for f in SQUAD_FEATURES]
EA_ALL = HOME_SQUAD + AWAY_SQUAD + DIFF_FEATURES
ALL_FEATURES = BASE_FEATURES + HOME_SQUAD + AWAY_SQUAD + DIFF_FEATURES
COMBINED_LEAN = BASE_FEATURES + DIFF_FEATURES


def join_squad_features(match_df, team_features):
    df = match_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = df['date'].dt.year

    available_years = sorted(team_features['year'].unique())

    def get_fifa_year(match_year):
        candidates = [y for y in available_years if y <= match_year]
        return max(candidates) if candidates else None

    df['fifa_year'] = df['match_year'].apply(get_fifa_year)

    home_tf = team_features.rename(columns={f: f'home_{f}' for f in SQUAD_FEATURES})
    home_tf = home_tf.rename(columns={'team': 'home_team', 'year': 'fifa_year'})
    away_tf = team_features.rename(columns={f: f'away_{f}' for f in SQUAD_FEATURES})
    away_tf = away_tf.rename(columns={'team': 'away_team', 'year': 'fifa_year'})

    home_cols = ['home_team', 'fifa_year'] + [f'home_{f}' for f in SQUAD_FEATURES]
    away_cols = ['away_team', 'fifa_year'] + [f'away_{f}' for f in SQUAD_FEATURES]

    df = df.merge(home_tf[home_cols], on=['home_team', 'fifa_year'], how='left')
    df = df.merge(away_tf[away_cols], on=['away_team', 'fifa_year'], how='left')

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
    print("EXP-11c: Retrain with enriched squad data (EA + FM23)")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_dc.csv")
    team_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Team features: {team_features.shape} ({team_features['team'].nunique()} teams)")

    # Join squad features
    print("\nJoining squad features...")
    train_aug = join_squad_features(train_df, team_features)
    test_aug = join_squad_features(test_df, team_features)

    train_has = train_aug['home_squad_avg_overall'].notna() & train_aug['away_squad_avg_overall'].notna()
    test_has = test_aug['home_squad_avg_overall'].notna() & test_aug['away_squad_avg_overall'].notna()

    print(f"Train coverage: {train_has.sum()}/{len(train_aug)} ({train_has.mean()*100:.1f}%)")
    print(f"Test coverage: {test_has.sum()}/{len(test_aug)} ({test_has.mean()*100:.1f}%)")

    train_sq = train_aug[train_has].copy()
    test_sq = test_aug[test_has].copy()

    # Labels
    le = LabelEncoder()
    y_train_all = le.fit_transform(train_aug['outcome'].values)
    y_test_all = le.transform(test_aug['outcome'].values)
    y_train_sq = le.transform(train_sq['outcome'].values)
    y_test_sq = le.transform(test_sq['outcome'].values)

    # DC model
    dc_bundle = joblib.load(MODELS_DIR / "dc_model.pkl")
    attack_params = dc_bundle["attack"]
    defense_params = dc_bundle["defense"]
    team_idx_dc = dc_bundle["team_idx"]
    with open(MODELS_DIR / "dc_params.json") as f:
        dp = json.load(f)
    home_adv_dc, rho_dc = dp["home_adv"], dp["rho"]

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
        aw = max(0, 1.0-hw-dr)
        t = hw+dr+aw
        return np.array([aw/t, dr/t, hw/t])

    dc_test_full = np.array([dc_probs(r['home_team'], r['away_team'],
        bool(r.get('neutral', r.get('neutral.1', True)))) for _, r in test_aug.iterrows()])
    dc_test_sq = np.array([dc_probs(r['home_team'], r['away_team'],
        bool(r.get('neutral', r.get('neutral.1', True)))) for _, r in test_sq.iterrows()])

    results = []
    XGB_P = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)

    def evaluate(name, y_true, y_proba, dc_proba=None, w_dc=1):
        if dc_proba is not None:
            proba = (5 * y_proba + w_dc * dc_proba) / (5 + w_dc)
        else:
            proba = y_proba
        ll = log_loss(y_true, proba)
        acc = accuracy_score(y_true, np.argmax(proba, axis=1))
        f1 = f1_score(y_true, np.argmax(proba, axis=1), average='macro')
        results.append((name, ll, acc, f1, len(y_true)))
        print(f"  {name:60s}  ll={ll:.4f}  acc={acc:.4f}  n={len(y_true)}")
        return ll

    # ═══════════════════════════════════════════════════
    # MODEL A: EA-only features
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A: EA Squad Features ONLY (squad-covered matches)")
    print("=" * 70)

    # A1: 9 diff features
    print("\n[A1] EA diff only (9 feat)")
    X_tr = train_sq[DIFF_FEATURES].fillna(0).values
    X_te = test_sq[DIFF_FEATURES].fillna(0).values
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    xgb_a1 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_a1.fit(X_tr, y_train_sq)
    p = xgb_a1.predict_proba(X_te)
    evaluate("A1: EA diff only (9 feat)", y_test_sq, p)
    evaluate("A1: EA diff only + DC blend", y_test_sq, p, dc_test_sq)

    # A2: All 53 EA features
    print("\n[A2] All EA features (53 feat)")
    X_tr = train_sq[EA_ALL].fillna(0).values
    X_te = test_sq[EA_ALL].fillna(0).values
    sc_a2 = StandardScaler()
    X_tr = sc_a2.fit_transform(X_tr)
    X_te = sc_a2.transform(X_te)
    xgb_a2 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_a2.fit(X_tr, y_train_sq)
    p_a2 = xgb_a2.predict_proba(X_te)
    evaluate("A2: All EA features (53 feat)", y_test_sq, p_a2)
    evaluate("A2: All EA features + DC blend", y_test_sq, p_a2, dc_test_sq)

    # A3: LR sanity check
    print("\n[A3] LR on EA diff")
    X_tr = train_sq[DIFF_FEATURES].fillna(0).values
    X_te = test_sq[DIFF_FEATURES].fillna(0).values
    sc3 = StandardScaler()
    X_tr = sc3.fit_transform(X_tr)
    X_te = sc3.transform(X_te)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial", random_state=42)
    lr.fit(X_tr, y_train_sq)
    p = lr.predict_proba(X_te)
    evaluate("A3: LR on EA diff (9 feat)", y_test_sq, p)
    evaluate("A3: LR on EA diff + DC blend", y_test_sq, p, dc_test_sq)

    # ═══════════════════════════════════════════════════
    # MODEL B: Baseline (no EA)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B: BASELINE (59 features, no EA)")
    print("=" * 70)

    # B1: 59 feat, 35K train, full test
    print("\n[B1] Baseline: 59 feat, 35K train, full test")
    X_tr = train_aug[BASE_FEATURES].values
    X_te = test_aug[BASE_FEATURES].values
    sc_b1 = StandardScaler()
    X_tr = sc_b1.fit_transform(X_tr)
    X_te = sc_b1.transform(X_te)
    ens_b1 = VotingClassifier(
        estimators=[
            ("xgb", CalibratedClassifierCV(xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0), method="isotonic", cv=5)),
            ("rf", CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1), method="isotonic", cv=5)),
        ], voting="soft", weights=[4, 1], n_jobs=-1)
    ens_b1.fit(X_tr, y_train_all)
    p_b1 = ens_b1.predict_proba(X_te)
    evaluate("B1: Baseline 59 feat, full test", y_test_all, p_b1)
    evaluate("B1: Baseline 59 feat + DC blend", y_test_all, p_b1, dc_test_full)

    # ═══════════════════════════════════════════════════
    # MODEL C: Combined (base + EA)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL C: COMBINED (Base + EA) — XGBoost NaN handling")
    print("=" * 70)

    # C1: XGB 112 feat, NaN native
    print("\n[C1] XGB 112 feat, 35K train, NaN native")
    X_tr_c1 = train_aug[ALL_FEATURES].values
    X_te_c1 = test_aug[ALL_FEATURES].values
    xgb_c1 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_c1.fit(X_tr_c1, y_train_all)
    p_c1 = xgb_c1.predict_proba(X_te_c1)
    evaluate("C1: XGB 112 feat, NaN native", y_test_all, p_c1)
    evaluate("C1: XGB 112 feat + DC blend", y_test_all, p_c1, dc_test_full)

    # C2: HistGBT 112 feat
    print("\n[C2] HistGBT 112 feat, 35K train")
    hgb_c2 = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, random_state=42),
        method="isotonic", cv=5)
    hgb_c2.fit(X_tr_c1, y_train_all)
    p_c2 = hgb_c2.predict_proba(X_te_c1)
    evaluate("C2: HistGBT 112 feat, NaN native", y_test_all, p_c2)
    evaluate("C2: HistGBT 112 feat + DC blend", y_test_all, p_c2, dc_test_full)

    # C3: Blend XGB + HistGBT
    print("\n[C3] Blend: XGB×4 + HistGBT×1")
    p_c3 = (4*p_c1 + 1*p_c2) / 5
    evaluate("C3: XGB×4 + HistGBT×1, 112 feat", y_test_all, p_c3)
    evaluate("C3: XGB×4 + HistGBT×1 + DC", y_test_all, p_c3, dc_test_full)

    # C4: Base + diff only (68 feat)
    print("\n[C4] XGB 68 feat (base+diff), NaN native")
    X_tr_c4 = train_aug[COMBINED_LEAN].values
    X_te_c4 = test_aug[COMBINED_LEAN].values
    xgb_c4 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_c4.fit(X_tr_c4, y_train_all)
    p_c4 = xgb_c4.predict_proba(X_te_c4)
    evaluate("C4: XGB 68 feat (base+diff)", y_test_all, p_c4)
    evaluate("C4: XGB 68 feat + DC blend", y_test_all, p_c4, dc_test_full)

    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ALL RESULTS (sorted by log loss)")
    print("=" * 80)
    print(f"  {'Model':60s}  {'LL':>8s}  {'Acc':>7s}  {'N':>5s}")
    print("-" * 90)
    for name, ll, acc, f1, n in sorted(results, key=lambda x: x[1]):
        print(f"  {name:60s}  {ll:8.4f}  {acc:7.4f}  {n:5d}")

    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    baseline_full = [r for r in results if r[0].startswith("B") and "full" in r[0] and "DC" in r[0]]
    combined_best = min([r for r in results if r[0].startswith("C")], key=lambda x: x[1])
    if baseline_full:
        bl = baseline_full[0]
        print(f"  Baseline (Phase 1):  {bl[0]:50s}  ll={bl[1]:.4f}")
    print(f"  Best combined:       {combined_best[0]:50s}  ll={combined_best[1]:.4f}")
    if baseline_full:
        delta = combined_best[1] - baseline_full[0][1]
        print(f"  Delta:               {delta:+.4f}")
        print(f"  Our Phase 1 best was 0.7988 (different test eval)")

    # Feature importance for best model
    print("\n\nTop 30 Feature Importance (C1: XGB 112 feat):")
    xgb_inner = xgb_c1.calibrated_classifiers_[0].estimator
    if hasattr(xgb_inner, 'feature_importances_'):
        imp = xgb_inner.feature_importances_
        feat_imp = sorted(zip(ALL_FEATURES, imp), key=lambda x: -x[1])
        for i, (f, v) in enumerate(feat_imp[:30]):
            is_ea = any(f.startswith(p) for p in ('home_squad','away_squad','squad_','home_def','away_def',
                'home_mid','away_mid','home_fwd','away_fwd','home_gk','away_gk','home_team_','away_team_',
                'home_strongest','away_strongest','home_weakest','away_weakest','def_avg','mid_avg',
                'fwd_avg','team_shooting','team_passing','team_defending'))
            marker = " ★ EA" if is_ea else ""
            print(f"  {i+1:3d}. {f:50s}  {v:.4f}{marker}")

    # Sanity check: 2026 WC matchups with COMBINED model
    print("\n" + "=" * 70)
    print("2026 WC SANITY CHECK — Combined model (C1) predictions")
    print("=" * 70)

    t26 = team_features[team_features['year'] == 2026].set_index('team')
    matchups = [
        ("Brazil", "Mexico", True),
        ("France", "Norway", True),
        ("Portugal", "Mexico", True),
        ("Spain", "Japan", True),
        ("Argentina", "Saudi Arabia", True),
        ("Germany", "South Korea", True),
        ("England", "United States", True),
        ("Belgium", "Morocco", True),
    ]

    print(f"\n  {'Home':15s} vs {'Away':15s}  {'C1 H':>7s}  {'C1 D':>7s}  {'C1 A':>7s}  | {'DC H':>6s}  {'DC D':>6s}  {'DC A':>6s}")
    print("-" * 85)

    for home, away, neutral in matchups:
        if home not in t26.index or away not in t26.index:
            print(f"  {home:15s} vs {away:15s}  — missing squad data")
            continue

        # Build full 112-feature vector (base features = 0/NaN since we don't have match context)
        feat_row = {f: 0 for f in BASE_FEATURES}
        # Set neutral
        feat_row['neutral.1'] = 1.0 if neutral else 0.0
        feat_row['tournament_importance'] = 4.0  # WC

        for f in SQUAD_FEATURES:
            feat_row[f'home_{f}'] = t26.loc[home, f]
            feat_row[f'away_{f}'] = t26.loc[away, f]

        feat_row['squad_avg_overall_diff'] = feat_row['home_squad_avg_overall'] - feat_row['away_squad_avg_overall']
        feat_row['squad_top3_avg_diff'] = feat_row['home_squad_top3_avg'] - feat_row['away_squad_top3_avg']
        feat_row['squad_value_diff'] = feat_row['home_squad_total_value'] - feat_row['away_squad_total_value']
        feat_row['def_avg_diff'] = feat_row['home_def_avg'] - feat_row['away_def_avg']
        feat_row['mid_avg_diff'] = feat_row['home_mid_avg'] - feat_row['away_mid_avg']
        feat_row['fwd_avg_diff'] = feat_row['home_fwd_avg'] - feat_row['away_fwd_avg']
        feat_row['team_shooting_diff'] = feat_row['home_team_shooting'] - feat_row['away_team_shooting']
        feat_row['team_passing_diff'] = feat_row['home_team_passing'] - feat_row['away_team_passing']
        feat_row['team_defending_diff'] = feat_row['home_team_defending'] - feat_row['away_team_defending']

        x = np.array([[feat_row.get(f, np.nan) for f in ALL_FEATURES]])
        p = xgb_c1.predict_proba(x)[0]
        dc_p = dc_probs(home, away, neutral)

        print(f"  {home:15s} vs {away:15s}  {p[2]:7.1%}  {p[1]:7.1%}  {p[0]:7.1%}  | {dc_p[2]:6.1%}  {dc_p[1]:6.1%}  {dc_p[0]:6.1%}")

    # Save best model
    best = min([r for r in results if r[0].startswith("C")], key=lambda x: x[1])
    print(f"\n\nBest model: {best[0]} (ll={best[1]:.4f})")

    if "C1" in best[0] or "C3" in best[0]:
        print("Saving C1 (XGB) model artifacts...")
        bundle = {
            "xgb": xgb_c1,
            "features": ALL_FEATURES,
            "label_encoder": le,
        }
        joblib.dump(bundle, MODELS_DIR / "best_model_exp11c.pkl")
        joblib.dump(ALL_FEATURES, MODELS_DIR / "feature_cols_exp11c.pkl")
        print("Saved!")


if __name__ == "__main__":
    main()
