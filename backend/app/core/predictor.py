"""
Match predictor — replicates notebook 07 exactly.
XGB×4 + RF×1 + DC×1 blended ensemble with 59-feature vector.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

ALL_CONFS = ["UEFA", "CAF", "AFC", "CONCACAF", "CONMEBOL", "OFC", "UNKNOWN"]

# ── Load models ──────────────────────────────────────────────
_bundle = joblib.load(MODELS_DIR / "best_model.pkl")
xgb_model = _bundle["xgb"]
rf_model = _bundle["rf"]
W_XGB = _bundle["w_xgb"]   # 4
W_RF = _bundle["w_rf"]     # 1
W_DC = 1                    # overridden from 5 → 1 (EXP-27)

scaler = joblib.load(MODELS_DIR / "scaler_dc.pkl")
le = joblib.load(MODELS_DIR / "label_encoder_dc.pkl")
FEATURE_COLS = joblib.load(MODELS_DIR / "feature_cols_dc.pkl")

dc_bundle = joblib.load(MODELS_DIR / "dc_model.pkl")
attack_params = dc_bundle["attack"]
defense_params = dc_bundle["defense"]
team_idx_dc = dc_bundle["team_idx"]

with open(MODELS_DIR / "dc_params.json") as f:
    _dc_params = json.load(f)
home_adv_dc = _dc_params["home_adv"]
rho_dc = _dc_params["rho"]

MAX_GOALS = 10

# ── Load data ────────────────────────────────────────────────
elos = pd.read_csv(DATA_DIR / "final_elos.csv").set_index("team")["final_elo"].to_dict()
confs = pd.read_csv(DATA_DIR / "team_confederations.csv").set_index("team")["confederation"].to_dict()

fm = pd.read_csv(DATA_DIR / "features_matrix.csv", parse_dates=["date"]).sort_values("date")

# Form lookups — latest row per team as home / away
_home_form_cols = [
    "home_win_rate_5", "home_avg_scored_5", "home_avg_conceded_5",
    "home_pts_per_match_5", "home_matches_played_5",
    "home_win_rate_10", "home_avg_scored_10", "home_avg_conceded_10",
    "home_pts_per_match_10", "home_matches_played_10",
]
_away_form_cols = [
    "away_win_rate_5", "away_avg_scored_5", "away_avg_conceded_5",
    "away_pts_per_match_5", "away_matches_played_5",
    "away_win_rate_10", "away_avg_scored_10", "away_avg_conceded_10",
    "away_pts_per_match_10", "away_matches_played_10",
]
latest_home = fm.groupby("home_team")[_home_form_cols].last()
latest_away = fm.groupby("away_team")[_away_form_cols].last()

# H2H lookup
_h2h_cols = [
    "h2h_home_win_rate", "h2h_home_avg_scored",
    "h2h_home_avg_conceded", "h2h_total_meetings", "h2h_recent_win_rate",
]
h2h_lookup = fm.groupby(["home_team", "away_team"])[_h2h_cols].last().to_dict("index")


# ── Helper functions ─────────────────────────────────────────

def conf_onehot(team: str) -> dict:
    conf = confs.get(team, "UNKNOWN")
    return {f"conf_{c}": int(conf == c) for c in ALL_CONFS}


def get_form(team: str) -> dict:
    if team in latest_home.index:
        row = latest_home.loc[team]
        return {
            "win_rate_5": row.get("home_win_rate_5", 0.5),
            "avg_scored_5": row.get("home_avg_scored_5", 1.3),
            "avg_conceded_5": row.get("home_avg_conceded_5", 1.0),
            "pts_per_match_5": row.get("home_pts_per_match_5", 1.5),
            "matches_played_5": row.get("home_matches_played_5", 5),
            "win_rate_10": row.get("home_win_rate_10", 0.5),
            "avg_scored_10": row.get("home_avg_scored_10", 1.3),
            "avg_conceded_10": row.get("home_avg_conceded_10", 1.0),
            "pts_per_match_10": row.get("home_pts_per_match_10", 1.5),
            "matches_played_10": row.get("home_matches_played_10", 10),
        }
    if team in latest_away.index:
        row = latest_away.loc[team]
        return {
            "win_rate_5": row.get("away_win_rate_5", 0.5),
            "avg_scored_5": row.get("away_avg_scored_5", 1.0),
            "avg_conceded_5": row.get("away_avg_conceded_5", 1.3),
            "pts_per_match_5": row.get("away_pts_per_match_5", 1.2),
            "matches_played_5": row.get("away_matches_played_5", 5),
            "win_rate_10": row.get("away_win_rate_10", 0.5),
            "avg_scored_10": row.get("away_avg_scored_10", 1.0),
            "avg_conceded_10": row.get("away_avg_conceded_10", 1.3),
            "pts_per_match_10": row.get("away_pts_per_match_10", 1.2),
            "matches_played_10": row.get("away_matches_played_10", 10),
        }
    return {
        "win_rate_5": 0.5, "avg_scored_5": 1.3, "avg_conceded_5": 1.3,
        "pts_per_match_5": 1.5, "matches_played_5": 5,
        "win_rate_10": 0.5, "avg_scored_10": 1.3, "avg_conceded_10": 1.3,
        "pts_per_match_10": 1.5, "matches_played_10": 10,
    }


def get_h2h(home: str, away: str) -> tuple:
    if (home, away) in h2h_lookup:
        d = h2h_lookup[(home, away)]
        return (
            d["h2h_home_win_rate"], d["h2h_home_avg_scored"],
            d["h2h_home_avg_conceded"], d["h2h_total_meetings"],
            d["h2h_recent_win_rate"],
        )
    return 0.5, 1.3, 1.3, 0, 0.5


# ── Dixon-Coles ──────────────────────────────────────────────

def dc_match_probs(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """Dixon-Coles win/draw/loss probabilities."""
    if home not in team_idx_dc or away not in team_idx_dc:
        return 0.45, 0.25, 0.30

    goals = np.arange(MAX_GOALS + 1)
    home_mask = goals[:, None] > goals[None, :]
    draw_mask = goals[:, None] == goals[None, :]
    away_mask = goals[:, None] < goals[None, :]

    hi, ai = team_idx_dc[home], team_idx_dc[away]
    ha = home_adv_dc if not neutral else 0.0
    lam = np.exp(attack_params[hi] + defense_params[ai] + ha)
    mu = np.exp(attack_params[ai] + defense_params[hi])

    score_mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    score_mat[0, 0] *= max(1 - lam * mu * rho_dc, 1e-10)
    score_mat[0, 1] *= max(1 + lam * rho_dc, 1e-10)
    score_mat[1, 0] *= max(1 + mu * rho_dc, 1e-10)
    score_mat[1, 1] *= max(1 - rho_dc, 1e-10)

    hw = float(np.sum(score_mat * home_mask))
    dr = float(np.sum(score_mat * draw_mask))
    aw = float(np.sum(score_mat * away_mask))
    total = hw + dr + aw
    return hw / total, dr / total, aw / total


# ── Feature builder ──────────────────────────────────────────

def build_features(home: str, away: str, neutral: bool = True) -> np.ndarray:
    """Build the 59-feature vector for a match."""
    home_elo = elos.get(home, 1500)
    away_elo = elos.get(away, 1500)
    elo_diff = home_elo - away_elo

    hf = get_form(home)
    af = get_form(away)
    h2h_hw, h2h_hs, h2h_hc, h2h_tm, h2h_rw = get_h2h(home, away)

    hc = conf_onehot(home)
    ac = conf_onehot(away)
    same_conf = int(confs.get(home, "X") == confs.get(away, "Y"))

    # Engineered features
    elo_diff_sq = elo_diff ** 2 * np.sign(elo_diff)
    home_form_momentum = hf["win_rate_5"] - hf["win_rate_10"]
    away_form_momentum = af["win_rate_5"] - af["win_rate_10"]
    home_gdf = hf["avg_scored_5"] - hf["avg_conceded_5"]
    away_gdf = af["avg_scored_5"] - af["avg_conceded_5"]
    net_goal_diff = home_gdf - away_gdf
    h2h_confidence = h2h_rw * (h2h_tm / (h2h_tm + 5))

    # DC features
    dc_hw, dc_dr, dc_aw = dc_match_probs(home, away, neutral)
    hi = team_idx_dc.get(home, 0)
    ai = team_idx_dc.get(away, 0)
    ha = home_adv_dc if not neutral else 0.0
    dc_lam = np.exp(attack_params[hi] + defense_params[ai] + ha)
    dc_mu = np.exp(attack_params[ai] + defense_params[hi])

    row = [
        home_elo, away_elo, elo_diff,
        hf["win_rate_5"], hf["avg_scored_5"], hf["avg_conceded_5"],
        hf["pts_per_match_5"], hf["matches_played_5"],
        hf["win_rate_10"], hf["avg_scored_10"], hf["avg_conceded_10"],
        hf["pts_per_match_10"], hf["matches_played_10"],
        af["win_rate_5"], af["avg_scored_5"], af["avg_conceded_5"],
        af["pts_per_match_5"], af["matches_played_5"],
        af["win_rate_10"], af["avg_scored_10"], af["avg_conceded_10"],
        af["pts_per_match_10"], af["matches_played_10"],
        h2h_hw, h2h_hs, h2h_hc, h2h_tm, h2h_rw,
        int(neutral), 60,  # neutral, tournament_importance
        hc["conf_UEFA"], hc["conf_CAF"], hc["conf_AFC"],
        hc["conf_CONCACAF"], hc["conf_CONMEBOL"], hc["conf_OFC"], hc["conf_UNKNOWN"],
        ac["conf_UEFA"], ac["conf_CAF"], ac["conf_AFC"],
        ac["conf_CONCACAF"], ac["conf_CONMEBOL"], ac["conf_OFC"], ac["conf_UNKNOWN"],
        same_conf,
        elo_diff_sq, home_form_momentum, away_form_momentum,
        home_gdf, away_gdf, net_goal_diff, h2h_confidence,
        dc_hw, dc_dr, dc_aw,
        dc_lam, dc_mu, dc_lam + dc_mu, dc_lam - dc_mu,
    ]
    return np.array(row, dtype=float)


# ── Predict match ────────────────────────────────────────────

def predict_match(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """
    Returns (p_home_win, p_draw, p_away_win).
    XGB×4 + RF×1 + DC×1 blend.
    """
    feat = build_features(home, away, neutral).reshape(1, -1)
    feat_s = scaler.transform(feat)

    # XGB and RF output order: [away_win, draw, home_win]
    xgb_p = xgb_model.predict_proba(feat_s)[0]
    rf_p = rf_model.predict_proba(feat_s)[0]

    # DC probs reordered to [away_win, draw, home_win]
    dc_hw, dc_dr, dc_aw = dc_match_probs(home, away, neutral)
    dc_p = np.array([dc_aw, dc_dr, dc_hw])

    blended = (W_XGB * xgb_p + W_RF * rf_p + W_DC * dc_p) / (W_XGB + W_RF + W_DC)

    # le.classes_ = ['away_win', 'draw', 'home_win']
    p_away = float(blended[0])
    p_draw = float(blended[1])
    p_home = float(blended[2])
    return p_home, p_draw, p_away


# ── Pre-cache all 48-team matchups ───────────────────────────

teams_2026 = pd.read_csv(DATA_DIR / "teams_2026.csv")
ALL_WC_TEAMS = teams_2026["team"].tolist()

prob_cache: dict[tuple[str, str], tuple[float, float, float]] = {}


def warm_cache():
    """Pre-compute all pairwise probabilities for the 48 WC teams."""
    for home in ALL_WC_TEAMS:
        for away in ALL_WC_TEAMS:
            if home != away:
                prob_cache[(home, away)] = predict_match(home, away, neutral=True)


def predict_match_cached(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """Use cache for WC teams, fall back to live prediction."""
    if (home, away) in prob_cache:
        return prob_cache[(home, away)]
    return predict_match(home, away, neutral)
