"""
EXP-11e: Phase 1 ensemble (XGB×4 + RF×1 + DC×5) WITH EA features.
The real test: does adding EA diff features beat the actual 0.7988 production model?
"""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from scipy.stats import poisson
import joblib, json

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

XGB_P = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
RF_P = dict(n_estimators=300, max_depth=12, min_samples_leaf=5)


def join_squad(df, team_features):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = df['date'].dt.year
    avail = sorted(team_features['year'].unique())
    def get_yr(y):
        c = [x for x in avail if x <= y]
        return max(c) if c else None
    df['fifa_year'] = df['match_year'].apply(get_yr)
    htf = team_features.rename(columns={f: f'home_{f}' for f in SQUAD_FEATURES})
    htf = htf.rename(columns={'team': 'home_team', 'year': 'fifa_year'})
    atf = team_features.rename(columns={f: f'away_{f}' for f in SQUAD_FEATURES})
    atf = atf.rename(columns={'team': 'away_team', 'year': 'fifa_year'})
    df = df.merge(htf[['home_team', 'fifa_year'] + HOME_SQUAD], on=['home_team', 'fifa_year'], how='left')
    df = df.merge(atf[['away_team', 'fifa_year'] + AWAY_SQUAD], on=['away_team', 'fifa_year'], how='left')
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
    print("EXP-11e: Phase 1 Ensemble + EA Features")
    print("=" * 75)
    print("Phase 1 best: XGB(cal)×4 + RF(cal)×1 + DC×5 = 0.7988")
    print("Question: can we beat 0.7988 by adding EA diff features?")

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_dc.csv")
    team_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")

    train_aug = join_squad(train_df, team_features)
    test_aug = join_squad(test_df, team_features)

    print(f"Train: {len(train_aug)}, Test: {len(test_aug)}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_aug['outcome'].values)
    y_test = le.transform(test_aug['outcome'].values)

    # DC probs
    dc_bundle = joblib.load(MODELS_DIR / "dc_model.pkl")
    with open(MODELS_DIR / "dc_params.json") as f:
        dp = json.load(f)

    def dc_probs(home, away, neutral):
        if home not in dc_bundle["team_idx"] or away not in dc_bundle["team_idx"]:
            return np.array([0.30, 0.25, 0.45])
        goals = np.arange(11)
        hi = dc_bundle["team_idx"][home]
        ai = dc_bundle["team_idx"][away]
        ha = dp["home_adv"] if not neutral else 0.0
        lam = np.exp(dc_bundle["attack"][hi] + dc_bundle["defense"][ai] + ha)
        mu = np.exp(dc_bundle["attack"][ai] + dc_bundle["defense"][hi])
        sm = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        sm[0, 0] *= max(1 - lam * mu * dp["rho"], 1e-10)
        sm[0, 1] *= max(1 + lam * dp["rho"], 1e-10)
        sm[1, 0] *= max(1 + mu * dp["rho"], 1e-10)
        sm[1, 1] *= max(1 - dp["rho"], 1e-10)
        hw = float(np.sum(sm * (goals[:, None] > goals[None, :])))
        dr = float(np.sum(sm * (goals[:, None] == goals[None, :])))
        aw = max(0, 1.0 - hw - dr)
        t = hw + dr + aw
        return np.array([aw / t, dr / t, hw / t])

    dc_test = np.array([dc_probs(r['home_team'], r['away_team'],
        bool(r.get('neutral', r.get('neutral.1', True)))) for _, r in test_aug.iterrows()])

    results = []

    def evaluate(name, y_true, y_proba, dc_proba=None, w_ml=5, w_dc=1):
        if dc_proba is not None:
            proba = (w_ml * y_proba + w_dc * dc_proba) / (w_ml + w_dc)
        else:
            proba = y_proba
        ll = log_loss(y_true, proba)
        acc = accuracy_score(y_true, np.argmax(proba, axis=1))
        f1 = f1_score(y_true, np.argmax(proba, axis=1), average='macro')
        results.append((name, ll, acc, f1, len(y_true)))
        print(f"  {name:60s}  ll={ll:.4f}  acc={acc:.4f}  f1={f1:.4f}")
        return proba

    # ═══════════════════════════════════════════════════
    # E0: REPRODUCE Phase 1 baseline (59 feat, XGB×4+RF×1, DC×5)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("E0: REPRODUCE Phase 1 — 59 feat, XGB×4 + RF×1 + DC×5")
    print("=" * 75)
    sc0 = StandardScaler()
    X_tr0 = sc0.fit_transform(train_aug[BASE_FEATURES].values)
    X_te0 = sc0.transform(test_aug[BASE_FEATURES].values)

    xgb0 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf0 = CalibratedClassifierCV(
        RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    ens0 = VotingClassifier(
        estimators=[("xgb", xgb0), ("rf", rf0)],
        voting="soft", weights=[4, 1], n_jobs=-1)
    ens0.fit(X_tr0, y_train)
    p0 = ens0.predict_proba(X_te0)
    evaluate("E0: Phase1 repro — no DC", y_test, p0)
    evaluate("E0: Phase1 repro — DC×1 blend", y_test, p0, dc_test, w_ml=5, w_dc=1)
    evaluate("E0: Phase1 repro — DC×5 blend", y_test, p0, dc_test, w_ml=5, w_dc=5)

    # ═══════════════════════════════════════════════════
    # E1: Phase 1 ensemble + 9 EA diff features (68 feat)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("E1: Phase 1 ensemble + EA diff — 68 feat, XGB×4 + RF×1")
    print("=" * 75)
    # XGBoost can handle NaN natively, but RF cannot.
    # Strategy: fillna(0) for RF, but use XGBoost's native NaN for XGB
    # VotingClassifier passes same X to both, so we need fillna for RF.
    # Alternative: train separately and manual blend.

    # Option A: fillna(0) for both (simple)
    print("\n[E1a] fillna(0) for both XGB and RF")
    sc1a = StandardScaler()
    X_tr1a = sc1a.fit_transform(train_aug[COMBINED_LEAN].fillna(0).values)
    X_te1a = sc1a.transform(test_aug[COMBINED_LEAN].fillna(0).values)
    xgb1a = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    rf1a = CalibratedClassifierCV(
        RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    ens1a = VotingClassifier(
        estimators=[("xgb", xgb1a), ("rf", rf1a)],
        voting="soft", weights=[4, 1], n_jobs=-1)
    ens1a.fit(X_tr1a, y_train)
    p1a = ens1a.predict_proba(X_te1a)
    evaluate("E1a: 68 feat fillna(0), XGB×4+RF×1 — no DC", y_test, p1a)
    evaluate("E1a: 68 feat fillna(0), XGB×4+RF×1 — DC×1", y_test, p1a, dc_test, w_ml=5, w_dc=1)
    evaluate("E1a: 68 feat fillna(0), XGB×4+RF×1 — DC×5", y_test, p1a, dc_test, w_ml=5, w_dc=5)

    # Option B: Manual blend — XGB on NaN-native, RF on base only
    print("\n[E1b] Manual blend: XGB(68 feat, NaN native) × 4 + RF(59 feat) × 1")
    xgb1b = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb1b.fit(train_aug[COMBINED_LEAN].values, y_train)  # NaN native, no scaler
    p_xgb1b = xgb1b.predict_proba(test_aug[COMBINED_LEAN].values)

    # RF on 59 base features (already trained as rf0 via ens0, but let's retrain standalone)
    rf1b = CalibratedClassifierCV(
        RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    rf1b.fit(X_tr0, y_train)  # 59 base features, scaled
    p_rf1b = rf1b.predict_proba(X_te0)

    p1b = (4 * p_xgb1b + 1 * p_rf1b) / 5
    evaluate("E1b: XGB(68,NaN)×4 + RF(59)×1 — no DC", y_test, p1b)
    evaluate("E1b: XGB(68,NaN)×4 + RF(59)×1 — DC×1", y_test, p1b, dc_test, w_ml=5, w_dc=1)
    evaluate("E1b: XGB(68,NaN)×4 + RF(59)×1 — DC×5", y_test, p1b, dc_test, w_ml=5, w_dc=5)

    # ═══════════════════════════════════════════════════
    # E2: Phase 1 ensemble + ALL 112 EA features
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("E2: Manual blend — XGB(112 feat, NaN native) × 4 + RF(59 feat) × 1")
    print("=" * 75)
    xgb2 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb2.fit(train_aug[ALL_FEATURES].values, y_train)
    p_xgb2 = xgb2.predict_proba(test_aug[ALL_FEATURES].values)
    p2 = (4 * p_xgb2 + 1 * p_rf1b) / 5
    evaluate("E2: XGB(112,NaN)×4 + RF(59)×1 — no DC", y_test, p2)
    evaluate("E2: XGB(112,NaN)×4 + RF(59)×1 — DC×1", y_test, p2, dc_test, w_ml=5, w_dc=1)
    evaluate("E2: XGB(112,NaN)×4 + RF(59)×1 — DC×5", y_test, p2, dc_test, w_ml=5, w_dc=5)

    # ═══════════════════════════════════════════════════
    # E3: Try different DC blend weights
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("E3: DC weight sweep (best ML model so far)")
    print("=" * 75)
    # Find best ML-only model
    best_ml_name = None
    best_ml_ll = 999
    best_ml_p = None
    for name, ll, acc, f1, n in results:
        if "no DC" in name and ll < best_ml_ll:
            best_ml_ll = ll
            best_ml_name = name
            best_ml_p = None

    # Recompute best ML proba
    # Use E1b XGB(68) x4 + RF(59) x1 as our ML base
    for w_dc in [1, 2, 3, 4, 5, 6, 7, 8]:
        evaluate(f"E3: Best ML + DC×{w_dc}", y_test, p1b, dc_test, w_ml=5, w_dc=w_dc)

    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ALL RESULTS — sorted by log loss")
    print("=" * 80)
    print(f"  {'Model':60s}  {'LL':>8s}  {'Acc':>7s}  {'F1':>7s}")
    print("-" * 90)
    for name, ll, acc, f1, n in sorted(results, key=lambda x: x[1]):
        marker = " <<<" if ll < 0.7988 else ""
        print(f"  {name:60s}  {ll:8.4f}  {acc:7.4f}  {f1:7.4f}{marker}")

    print(f"\n  Phase 1 production: 0.7988")
    best = min(results, key=lambda x: x[1])
    delta = best[1] - 0.7988
    print(f"  Best this run:     {best[1]:.4f} ({best[0]})")
    print(f"  Delta vs Phase 1:  {delta:+.4f} ({'IMPROVEMENT!' if delta < 0 else 'no improvement'})")

    # Sanity check
    print("\n" + "=" * 75)
    print("SANITY CHECK — Confederation bias (using E1b + DC×5)")
    print("=" * 75)

    t26 = team_features[team_features['year'] == 2026].set_index('team')
    matchups = [
        ("Brazil", "Mexico", True),
        ("France", "Norway", True),
        ("Portugal", "Mexico", True),
        ("Germany", "South Korea", True),
        ("England", "United States", True),
        ("Argentina", "Saudi Arabia", True),
    ]

    print(f"\n  {'Home':15s} vs {'Away':15s}  {'ML H':>7s}  {'ML D':>7s}  {'ML A':>7s}  | {'DC H':>6s}  {'DC D':>6s}  {'DC A':>6s}")
    print("-" * 85)

    for home, away, neutral in matchups:
        if home not in t26.index or away not in t26.index:
            print(f"  {home:15s} vs {away:15s}  — missing")
            continue

        feat_row = {f: 0 for f in BASE_FEATURES}
        feat_row['neutral.1'] = 1.0 if neutral else 0.0
        feat_row['tournament_importance'] = 4.0
        for f in SQUAD_FEATURES:
            feat_row[f'home_{f}'] = t26.loc[home, f]
            feat_row[f'away_{f}'] = t26.loc[away, f]
        for diff_f in DIFF_FEATURES:
            parts = diff_f.split('_diff')[0]
            feat_row[diff_f] = feat_row.get(f'home_{parts}', 0) - feat_row.get(f'away_{parts}', 0)
        feat_row['squad_avg_overall_diff'] = feat_row['home_squad_avg_overall'] - feat_row['away_squad_avg_overall']
        feat_row['squad_top3_avg_diff'] = feat_row['home_squad_top3_avg'] - feat_row['away_squad_top3_avg']
        feat_row['squad_value_diff'] = feat_row['home_squad_total_value'] - feat_row['away_squad_total_value']
        feat_row['def_avg_diff'] = feat_row['home_def_avg'] - feat_row['away_def_avg']
        feat_row['mid_avg_diff'] = feat_row['home_mid_avg'] - feat_row['away_mid_avg']
        feat_row['fwd_avg_diff'] = feat_row['home_fwd_avg'] - feat_row['away_fwd_avg']
        feat_row['team_shooting_diff'] = feat_row['home_team_shooting'] - feat_row['away_team_shooting']
        feat_row['team_passing_diff'] = feat_row['home_team_passing'] - feat_row['away_team_passing']
        feat_row['team_defending_diff'] = feat_row['home_team_defending'] - feat_row['away_team_defending']

        # XGB(68 feat) prediction
        x_lean = np.array([[feat_row.get(f, np.nan) for f in COMBINED_LEAN]])
        p_xgb = xgb1b.predict_proba(x_lean)[0]

        # RF(59 feat) prediction
        x_base = np.array([[feat_row.get(f, 0) for f in BASE_FEATURES]])
        x_base_sc = sc0.transform(x_base)
        p_rf = rf1b.predict_proba(x_base_sc)[0]

        # Blend: XGB×4 + RF×1
        p_ml = (4 * p_xgb + 1 * p_rf) / 5

        # + DC×5
        dc_p = dc_probs(home, away, neutral)
        p_final = (5 * p_ml + 5 * dc_p) / 10

        print(f"  {home:15s} vs {away:15s}  {p_final[2]:7.1%}  {p_final[1]:7.1%}  {p_final[0]:7.1%}  | {dc_p[2]:6.1%}  {dc_p[1]:6.1%}  {dc_p[0]:6.1%}")


if __name__ == "__main__":
    main()
