"""
Phase 3 — Train enriched ensemble with new B5 features.

Architecture: XGB(NaN-native)×4 + RF×1 + DC×1 blend (same as Phase 2 winner).

New features vs Phase 2:
  • SB team features (39/48 teams): xg_for_per_match, xg_against_per_match,
    def_overperformance, att_overperformance, set_piece_share, n_matches
  • International form (48/48): goals_for/against_per_match, win_rate,
    last_10 form, n_matches_2y
  • Chemistry (48/48): same_club_top1/3, n_unique_clubs, avg_caps, avg_age

Training window: 2018-01-01 → 2026-03-31  (modern era, ~8K matches)
Holdout: rolling 12 months (2025-06-11 → 2026-03-31)
Honesty: 0 WC 2026 matches in train OR test.

Compares against:
  1. Phase 2 architecture using same data (apples-to-apples)
  2. "Pick higher squad rating" baseline
  3. "Pick higher ELO" baseline
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Feature catalogs ────────────────────────────────────────────────────

SQUAD_FEATURES = [
    "squad_avg_overall", "squad_median_overall", "squad_std_overall",
    "squad_top3_avg", "squad_bottom5_avg",
    "gk_avg", "def_avg", "mid_avg", "fwd_avg",
    "strongest_unit", "weakest_unit",
    "squad_total_value", "squad_avg_value",
    "squad_avg_age", "squad_avg_potential_gap", "squad_avg_caps",
    "team_pace", "team_shooting", "team_passing",
    "team_dribbling", "team_defending", "team_physic",
]
DIFF_FEATURES = [
    "squad_avg_overall_diff", "squad_top3_avg_diff", "squad_value_diff",
    "def_avg_diff", "mid_avg_diff", "fwd_avg_diff",
    "team_shooting_diff", "team_passing_diff", "team_defending_diff",
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

# New Phase 3 features (current-snapshot, static across matches)
SB_FEATURES = [
    "sb_xg_for_per_match", "sb_xg_against_per_match",
    "sb_def_overperformance", "sb_att_overperformance",
    "sb_xg_set_piece_share", "sb_n_matches",
]
INTL_FEATURES = [
    "intl_goals_for_per_match", "intl_goals_against_per_match",
    "intl_win_rate", "intl_draw_rate", "intl_form_last10",
    "intl_n_matches_2y", "intl_competitive_pct",
]
CHEM_FEATURES = [
    "same_club_top1_pct", "same_club_top3_pct",
    "n_unique_clubs", "avg_intl_caps", "avg_squad_age",
]

# Diffs/sums for the new features (often more predictive than raw values)
NEW_DIFF_FEATURES = [
    "sb_xg_for_diff", "sb_def_overperf_diff", "sb_att_overperf_diff",
    "intl_form_diff", "intl_win_rate_diff",
    "intl_gf_diff", "intl_ga_diff",
    "chem_same_club_diff", "chem_caps_diff",
]

HOME_SQUAD = [f"home_{f}" for f in SQUAD_FEATURES]
AWAY_SQUAD = [f"away_{f}" for f in SQUAD_FEATURES]
HOME_SB = [f"home_{f}" for f in SB_FEATURES]
AWAY_SB = [f"away_{f}" for f in SB_FEATURES]
HOME_INTL = [f"home_{f}" for f in INTL_FEATURES]
AWAY_INTL = [f"away_{f}" for f in INTL_FEATURES]
HOME_CHEM = [f"home_{f}" for f in CHEM_FEATURES]
AWAY_CHEM = [f"away_{f}" for f in CHEM_FEATURES]

PHASE2_FEATURES = BASE_FEATURES + HOME_SQUAD + AWAY_SQUAD + DIFF_FEATURES
PHASE3_FEATURES = (PHASE2_FEATURES + HOME_SB + AWAY_SB + HOME_INTL + AWAY_INTL
                   + HOME_CHEM + AWAY_CHEM + NEW_DIFF_FEATURES)

# Hyperparameters (Phase 2 winners)
XGB_P = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
RF_P = dict(n_estimators=300, max_depth=12, min_samples_leaf=5)

# Holdout window
TRAIN_START = "2018-01-01"
HOLDOUT_START = "2025-06-11"   # 12 months before WC 2026 opening day
HOLDOUT_END = "2026-03-31"     # last date in matches_clean


def join_squad(df: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Join team_features_by_year on (team, fifa_year) — use most recent year ≤ match year."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["match_year"] = df["date"].dt.year
    avail = sorted(team_features["year"].unique())

    def get_year(y):
        c = [x for x in avail if x <= y]
        return max(c) if c else avail[0]

    df["fifa_year"] = df["match_year"].apply(get_year)

    htf = team_features.rename(columns={f: f"home_{f}" for f in SQUAD_FEATURES})
    htf = htf.rename(columns={"team": "home_team", "year": "fifa_year"})
    atf = team_features.rename(columns={f: f"away_{f}" for f in SQUAD_FEATURES})
    atf = atf.rename(columns={"team": "away_team", "year": "fifa_year"})

    df = df.merge(htf[["home_team", "fifa_year"] + HOME_SQUAD],
                  on=["home_team", "fifa_year"], how="left")
    df = df.merge(atf[["away_team", "fifa_year"] + AWAY_SQUAD],
                  on=["away_team", "fifa_year"], how="left")

    for f in ["squad_avg_overall", "squad_top3_avg", "squad_total_value",
              "def_avg", "mid_avg", "fwd_avg",
              "team_shooting", "team_passing", "team_defending"]:
        col = "squad_value_diff" if f == "squad_total_value" else f"{f}_diff"
        df[col] = df[f"home_{f}"] - df[f"away_{f}"]
    return df


def join_static(df: pd.DataFrame, sb: pd.DataFrame, intl: pd.DataFrame, chem: pd.DataFrame) -> pd.DataFrame:
    """Join the new B5 features by team (static current-snapshot)."""
    df = df.copy()
    h_sb = sb.rename(columns={f: f"home_{f}" for f in SB_FEATURES})
    h_sb = h_sb.rename(columns={"team": "home_team"})[["home_team"] + HOME_SB]
    a_sb = sb.rename(columns={f: f"away_{f}" for f in SB_FEATURES})
    a_sb = a_sb.rename(columns={"team": "away_team"})[["away_team"] + AWAY_SB]
    df = df.merge(h_sb, on="home_team", how="left").merge(a_sb, on="away_team", how="left")

    h_in = intl.rename(columns={f: f"home_{f}" for f in INTL_FEATURES})
    h_in = h_in.rename(columns={"team": "home_team"})[["home_team"] + HOME_INTL]
    a_in = intl.rename(columns={f: f"away_{f}" for f in INTL_FEATURES})
    a_in = a_in.rename(columns={"team": "away_team"})[["away_team"] + AWAY_INTL]
    df = df.merge(h_in, on="home_team", how="left").merge(a_in, on="away_team", how="left")

    h_ch = chem.rename(columns={f: f"home_{f}" for f in CHEM_FEATURES})
    h_ch = h_ch.rename(columns={"team": "home_team"})[["home_team"] + HOME_CHEM]
    a_ch = chem.rename(columns={f: f"away_{f}" for f in CHEM_FEATURES})
    a_ch = a_ch.rename(columns={"team": "away_team"})[["away_team"] + AWAY_CHEM]
    df = df.merge(h_ch, on="home_team", how="left").merge(a_ch, on="away_team", how="left")

    # Diffs
    df["sb_xg_for_diff"] = df["home_sb_xg_for_per_match"] - df["away_sb_xg_for_per_match"]
    df["sb_def_overperf_diff"] = df["home_sb_def_overperformance"] - df["away_sb_def_overperformance"]
    df["sb_att_overperf_diff"] = df["home_sb_att_overperformance"] - df["away_sb_att_overperformance"]
    df["intl_form_diff"] = df["home_intl_form_last10"] - df["away_intl_form_last10"]
    df["intl_win_rate_diff"] = df["home_intl_win_rate"] - df["away_intl_win_rate"]
    df["intl_gf_diff"] = df["home_intl_goals_for_per_match"] - df["away_intl_goals_for_per_match"]
    df["intl_ga_diff"] = df["home_intl_goals_against_per_match"] - df["away_intl_goals_against_per_match"]
    df["chem_same_club_diff"] = df["home_same_club_top3_pct"] - df["away_same_club_top3_pct"]
    df["chem_caps_diff"] = df["home_avg_intl_caps"] - df["away_avg_intl_caps"]
    return df


def evaluate(name: str, y_true, y_proba, results: list, baseline_class: int = 2) -> None:
    """Compute log loss, accuracy, Brier, RPS, ECE."""
    ll = log_loss(y_true, y_proba)
    acc = accuracy_score(y_true, np.argmax(y_proba, axis=1))

    # Brier score (multiclass: mean over per-class one-vs-rest brier)
    n_classes = y_proba.shape[1]
    y_oh = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((y_proba - y_oh) ** 2, axis=1))

    # RPS (Rank Probability Score for ordered outcomes: away_win < draw < home_win)
    cum_pred = np.cumsum(y_proba, axis=1)
    cum_true = np.cumsum(y_oh, axis=1)
    rps = np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1)) / (n_classes - 1)

    # ECE
    preds = np.argmax(y_proba, axis=1)
    confs = np.max(y_proba, axis=1)
    correct = (preds == y_true).astype(int)
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(y_true)) * abs(correct[m].mean() - confs[m].mean())

    results.append({"name": name, "log_loss": ll, "accuracy": acc,
                    "brier": brier, "rps": rps, "ece": ece, "n": len(y_true)})
    print(f"  {name:55s}  ll={ll:.4f}  acc={acc:.4f}  Brier={brier:.4f}  RPS={rps:.4f}  ECE={ece:.4f}")


def dc_probs_fn(dc_ratings: pd.DataFrame, dp: dict):
    rmap = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}

    def f(home, away, neutral):
        if home not in rmap or away not in rmap:
            return np.array([0.45, 0.25, 0.30])  # away_win, draw, home_win
        goals = np.arange(11)
        h_att, h_def = rmap[home]
        a_att, a_def = rmap[away]
        ha = dp["home_adv"] if not neutral else 0.0
        lam = np.exp(h_att + a_def + ha)
        mu = np.exp(a_att + h_def)
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
    return f


def main():
    print("=" * 80)
    print("PHASE 3 TRAINING — XGB+RF ensemble with new B5 features")
    print("=" * 80)

    # ── Load
    print("\n[Load] data files...")
    train_dc = pd.read_csv(PROC / "train_dc.csv", parse_dates=["date"])
    test_dc = pd.read_csv(PROC / "test_dc.csv", parse_dates=["date"])
    full = pd.concat([train_dc, test_dc], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"  full pool: {len(full):,} matches, {full['date'].min().date()} → {full['date'].max().date()}")

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    print(f"  team_features_by_year: {len(tfy)} rows")
    print(f"  SB features: {len(sb)} teams, intl: {len(intl)}, chem: {len(chem)}")

    # ── Filter to Phase 3 training window
    pool = full[(full["date"] >= TRAIN_START) & (full["date"] <= HOLDOUT_END)].copy()
    print(f"\n[Filter] modern era {TRAIN_START} → {HOLDOUT_END}: {len(pool):,} matches")

    # Train / test split — rolling 12-month holdout
    train = pool[pool["date"] < HOLDOUT_START].copy()
    test = pool[pool["date"] >= HOLDOUT_START].copy()
    print(f"  train: {len(train):,}, test (holdout): {len(test):,}")

    # ── Join features
    print("\n[Join] features...")
    train = join_squad(train, tfy)
    test = join_squad(test, tfy)
    train = join_static(train, sb, intl, chem)
    test = join_static(test, sb, intl, chem)
    print(f"  train cols: {len(train.columns)}, test cols: {len(test.columns)}")

    # ── Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train["outcome"].values)
    y_test = le.transform(test["outcome"].values)
    print(f"  classes (in order): {list(le.classes_)}")  # ['away_win','draw','home_win']

    # ── DC test probs (reconstruct from saved DC ratings + params)
    dc_ratings = pd.read_csv(PROC / "dc_ratings.csv")
    with open(MODELS / "dc_params.json") as f:
        dp = json.load(f)
    dc_fn = dc_probs_fn(dc_ratings, dp)
    print("\n[DC] computing DC probabilities for holdout...")
    dc_test = np.array([
        dc_fn(r["home_team"], r["away_team"],
              bool(r.get("neutral.1", r.get("neutral", True))))
        for _, r in test.iterrows()
    ])

    results = []

    # ════════════════════════════════════════════════════
    # M0: Phase 2 architecture on same train/test window (baseline)
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("M0: Phase 2 architecture — BASE + SQUAD + DIFF features (no B5 new)")
    print("=" * 80)
    feats_m0 = PHASE2_FEATURES

    sc = StandardScaler()
    X_tr_rf = sc.fit_transform(train[feats_m0].fillna(0).values)
    X_te_rf = sc.transform(test[feats_m0].fillna(0).values)

    xgb_m0 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_m0.fit(train[feats_m0].values, y_train)
    p_xgb_m0 = xgb_m0.predict_proba(test[feats_m0].values)

    rf_m0 = CalibratedClassifierCV(
        RandomForestClassifier(**RF_P, random_state=42, n_jobs=-1),
        method="isotonic", cv=5)
    rf_m0.fit(X_tr_rf, y_train)
    p_rf_m0 = rf_m0.predict_proba(X_te_rf)

    p_m0 = (4 * p_xgb_m0 + 1 * p_rf_m0) / 5
    evaluate("M0: Phase 2 arch — XGB×4 + RF×1, no DC", y_test, p_m0, results)
    p_m0_dc = (5 * p_m0 + 1 * dc_test) / 6
    evaluate("M0: Phase 2 arch — XGB×4 + RF×1 + DC×1", y_test, p_m0_dc, results)

    # ════════════════════════════════════════════════════
    # M1: Phase 3 — add SB, intl, chem features
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("M1: Phase 3 — BASE + SQUAD + DIFF + SB + INTL + CHEM features")
    print("=" * 80)
    feats_m1 = PHASE3_FEATURES

    xgb_m1 = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_P, eval_metric="mlogloss", random_state=42, verbosity=0),
        method="isotonic", cv=5)
    xgb_m1.fit(train[feats_m1].values, y_train)
    p_xgb_m1 = xgb_m1.predict_proba(test[feats_m1].values)

    # RF uses base only (handles NaN poorly)
    p_m1 = (4 * p_xgb_m1 + 1 * p_rf_m0) / 5
    evaluate("M1: Phase 3 — XGB(rich)×4 + RF(base)×1, no DC", y_test, p_m1, results)
    p_m1_dc = (5 * p_m1 + 1 * dc_test) / 6
    evaluate("M1: Phase 3 — XGB(rich)×4 + RF(base)×1 + DC×1", y_test, p_m1_dc, results)

    # ════════════════════════════════════════════════════
    # M2: XGB only (rich features) — feature importance reveal
    # ════════════════════════════════════════════════════
    p_m2_dc = (5 * p_xgb_m1 + 1 * dc_test) / 6
    evaluate("M2: Pure XGB(rich) + DC×1", y_test, p_xgb_m1, results)
    evaluate("M2: Pure XGB(rich) + DC×1 blend", y_test, p_m2_dc, results)

    # ════════════════════════════════════════════════════
    # Baselines
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Baselines (the bars we have to beat)")
    print("=" * 80)
    # Baseline 1: pick higher squad rating
    def baseline_squad(row):
        h = row["home_squad_avg_overall"]
        a = row["away_squad_avg_overall"]
        if pd.isna(h) or pd.isna(a):
            return np.array([0.45, 0.25, 0.30])  # default
        if h > a:
            return np.array([0.20, 0.20, 0.60])
        elif a > h:
            return np.array([0.60, 0.20, 0.20])
        else:
            return np.array([0.35, 0.30, 0.35])
    p_squad = np.array([baseline_squad(r) for _, r in test.iterrows()])
    evaluate("Baseline: pick higher squad rating", y_test, p_squad, results)

    # Baseline 2: pick higher ELO
    def baseline_elo(row):
        diff = row["elo_diff"]
        if diff > 50:
            return np.array([0.15, 0.20, 0.65])
        elif diff > 0:
            return np.array([0.30, 0.25, 0.45])
        elif diff > -50:
            return np.array([0.45, 0.25, 0.30])
        else:
            return np.array([0.65, 0.20, 0.15])
    p_elo = np.array([baseline_elo(r) for _, r in test.iterrows()])
    evaluate("Baseline: pick higher ELO", y_test, p_elo, results)

    # ════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    rdf = pd.DataFrame(results).round(4)
    print(rdf.to_string(index=False))

    # Save best model
    print("\n[Save] best model (M1: Phase 3 ensemble + DC×1)...")
    joblib.dump({
        "xgb": xgb_m1,
        "rf": rf_m0,
        "features_xgb": feats_m1,
        "features_rf": feats_m0,
        "label_encoder": le,
        "weights": {"xgb": 4, "rf": 1, "dc": 1},
    }, MODELS / "phase3_model.pkl")

    rdf.to_csv(PROC / "phase3_results.csv", index=False)
    print(f"  → models/phase3_model.pkl + data/processed/phase3_results.csv")


if __name__ == "__main__":
    main()
