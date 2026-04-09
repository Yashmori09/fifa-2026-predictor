"""
EXP-11d: Train on 2014+ data only (93% squad coverage).
Compare: does high coverage on 6.5K matches beat low coverage on 35K?
"""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
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
    print("EXP-11d: 2014+ Training Only (93% squad coverage)")
    print("=" * 75)

    train_df = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_dc.csv")
    team_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")

    train_aug = join_squad(train_df, team_features)
    test_aug = join_squad(test_df, team_features)

    # 2014+ train with squad coverage
    train_aug['date'] = pd.to_datetime(train_aug['date'])
    train_2014 = train_aug[train_aug['date'].dt.year >= 2014].copy()
    has_sq_tr = train_2014['home_squad_avg_overall'].notna() & train_2014['away_squad_avg_overall'].notna()
    train_sq = train_2014[has_sq_tr].copy()

    has_sq_te = test_aug['home_squad_avg_overall'].notna() & test_aug['away_squad_avg_overall'].notna()
    test_sq = test_aug[has_sq_te].copy()

    print(f"Train 2014+ w/ squad: {len(train_sq)}")
    print(f"Test w/ squad: {len(test_sq)}/{len(test_aug)}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_sq['outcome'].values)
    y_test = le.transform(test_sq['outcome'].values)
    y_train_all = le.transform(train_aug['outcome'].values)
    y_test_all = le.transform(test_aug['outcome'].values)

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

    dc_sq = np.array([dc_probs(r['home_team'], r['away_team'],
        bool(r.get('neutral', r.get('neutral.1', True)))) for _, r in test_sq.iterrows()])
    dc_full = np.array([dc_probs(r['home_team'], r['away_team'],
        bool(r.get('neutral', r.get('neutral.1', True)))) for _, r in test_aug.iterrows()])

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
        results.append((name, ll, acc, len(y_true)))
        print(f"  {name:55s}  ll={ll:.4f}  acc={acc:.4f}  n={len(y_true)}")

    # ═══════════════════════════════════════════════════
    # D1: 68 feat (base + diff), 2014+ train
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("D1: XGB 68 feat (base+diff), 6K train (2014+)")
    print("=" * 75)
    X_tr = train_sq[COMBINED_LEAN].fillna(0).values
    X_te = test_sq[COMBINED_LEAN].fillna(0).values
    sc1 = StandardScaler()
    X_tr = sc1.fit_transform(X_tr)
    X_te = sc1.transform(X_te)
    d1 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    d1.fit(X_tr, y_train)
    p1 = d1.predict_proba(X_te)
    evaluate("D1: 68 feat, 6K train", y_test, p1)
    evaluate("D1: 68 feat, 6K train + DC", y_test, p1, dc_sq)

    # ═══════════════════════════════════════════════════
    # D2: 112 feat (all), 2014+ train
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("D2: XGB 112 feat (all), 6K train (2014+)")
    print("=" * 75)
    X_tr = train_sq[ALL_FEATURES].fillna(0).values
    X_te = test_sq[ALL_FEATURES].fillna(0).values
    sc2 = StandardScaler()
    X_tr = sc2.fit_transform(X_tr)
    X_te = sc2.transform(X_te)
    d2 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    d2.fit(X_tr, y_train)
    p2 = d2.predict_proba(X_te)
    evaluate("D2: 112 feat, 6K train", y_test, p2)
    evaluate("D2: 112 feat, 6K train + DC", y_test, p2, dc_sq)

    # ═══════════════════════════════════════════════════
    # D3: 53 EA-only feat, 2014+ train
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("D3: XGB 53 EA-only feat, 6K train (2014+)")
    print("=" * 75)
    X_tr = train_sq[EA_ALL].fillna(0).values
    X_te = test_sq[EA_ALL].fillna(0).values
    sc3 = StandardScaler()
    X_tr = sc3.fit_transform(X_tr)
    X_te = sc3.transform(X_te)
    d3 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    d3.fit(X_tr, y_train)
    p3 = d3.predict_proba(X_te)
    evaluate("D3: 53 EA feat, 6K train", y_test, p3)
    evaluate("D3: 53 EA feat, 6K train + DC", y_test, p3, dc_sq)

    # ═══════════════════════════════════════════════════
    # BASELINES (fair comparison on same test set)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("BASELINES (same squad-covered test set for fair comparison)")
    print("=" * 75)

    # Baseline A: 35K train, 59 base feat, eval on squad test
    print("\n[BA] 35K train, 59 base feat")
    scba = StandardScaler()
    X_tr_ba = scba.fit_transform(train_aug[BASE_FEATURES].values)
    X_te_ba = scba.transform(test_sq[BASE_FEATURES].values)
    ba = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    ba.fit(X_tr_ba, y_train_all)
    pba = ba.predict_proba(X_te_ba)
    evaluate("BA: 59 feat, 35K train", y_test, pba)
    evaluate("BA: 59 feat, 35K train + DC", y_test, pba, dc_sq)

    # Baseline B: 6K train, 59 base feat
    print("\n[BB] 6K train (2014+), 59 base feat")
    scbb = StandardScaler()
    X_tr_bb = scbb.fit_transform(train_sq[BASE_FEATURES].values)
    X_te_bb = scbb.transform(test_sq[BASE_FEATURES].values)
    bb = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    bb.fit(X_tr_bb, y_train)
    pbb = bb.predict_proba(X_te_bb)
    evaluate("BB: 59 feat, 6K train (2014+)", y_test, pbb)
    evaluate("BB: 59 feat, 6K train + DC", y_test, pbb, dc_sq)

    # C4 equiv: 35K train, 68 feat (base+diff)
    print("\n[C4] 35K train, 68 feat (base+diff)")
    scc4 = StandardScaler()
    X_tr_c4 = scc4.fit_transform(train_aug[COMBINED_LEAN].values)
    X_te_c4 = scc4.transform(test_sq[COMBINED_LEAN].fillna(0).values)
    c4 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    c4.fit(X_tr_c4, y_train_all)
    pc4 = c4.predict_proba(X_te_c4)
    evaluate("C4: 68 feat, 35K train", y_test, pc4)
    evaluate("C4: 68 feat, 35K train + DC", y_test, pc4, dc_sq)

    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ALL RESULTS — sorted by log loss (all on same test set)")
    print("=" * 80)
    print(f"  {'Model':55s}  {'LL':>8s}  {'Acc':>7s}  {'N':>5s}")
    print("-" * 85)
    for name, ll, acc, n in sorted(results, key=lambda x: x[1]):
        print(f"  {name:55s}  {ll:8.4f}  {acc:7.4f}  {n:5d}")

    # Key comparison
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does 93% coverage on 6K beat 17% coverage on 35K?")
    print("=" * 80)
    best_6k = min([r for r in results if "6K" in r[0]], key=lambda x: x[1])
    best_35k = min([r for r in results if "35K" in r[0]], key=lambda x: x[1])
    print(f"  Best 6K (2014+):  {best_6k[0]:55s}  ll={best_6k[1]:.4f}")
    print(f"  Best 35K (all):   {best_35k[0]:55s}  ll={best_35k[1]:.4f}")
    delta = best_6k[1] - best_35k[1]
    print(f"  Delta: {delta:+.4f} ({'6K wins' if delta < 0 else '35K wins'})")


if __name__ == "__main__":
    main()
