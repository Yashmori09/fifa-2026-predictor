"""
Match predictor — Phase 2.
XGB×3 + RF×1 blended ensemble with 97-feature vector (EA + ELO, no DC blend).
Scorelines generated via reverse-engineered Poisson from model probabilities.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

ALL_CONFS = ["UEFA", "CAF", "AFC", "CONCACAF", "CONMEBOL", "OFC", "UNKNOWN"]

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

# ── Load Phase 2 model ─────────────────────────────────────
_bundle = joblib.load(MODELS_DIR / "phase2_model.pkl")
xgb_model = _bundle["xgb"]
rf_model = _bundle["rf"]
W_XGB = _bundle["w_xgb"]   # 3
W_RF = _bundle["w_rf"]     # 1
le = _bundle["label_encoder"]
FEATURE_COLS = _bundle["feature_cols"]

# ── Load data ──────────────────────────────────────────────
elos = pd.read_csv(DATA_DIR / "final_elos.csv").set_index("team")["final_elo"].to_dict()
confs = pd.read_csv(DATA_DIR / "team_confederations.csv").set_index("team")["confederation"].to_dict()

fm = pd.read_csv(DATA_DIR / "features_matrix.csv", parse_dates=["date"]).sort_values("date")

# Squad features by year
team_features = pd.read_csv(DATA_DIR / "team_features_by_year.csv")
_tf_avail_years = sorted(team_features['year'].unique())

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


# ── Helper functions ────────────────────────────────────────

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


# Pre-build squad feature lookup: {(team, year): {feat: val}}
_squad_lookup = {}
for _, _row in team_features.iterrows():
    _squad_lookup[(_row['team'], int(_row['year']))] = {f: _row[f] for f in SQUAD_FEATURES}

_NAN_SQUAD = {f: np.nan for f in SQUAD_FEATURES}


def get_squad_features(team: str, year: int = 2026) -> dict:
    """Get squad features for a team at a given year (latest available <= year)."""
    candidates = [y for y in _tf_avail_years if y <= year]
    if not candidates:
        return _NAN_SQUAD
    fifa_year = max(candidates)
    return _squad_lookup.get((team, fifa_year), _NAN_SQUAD)


# ── Reverse Poisson (precomputed lookup) ───────────────────

# Build a lookup table: for grid of (lam_h, lam_a) → (p_home, p_draw, p_away)
# Then use nearest-neighbor lookup instead of per-pair optimization.
from scipy.stats import poisson as _poisson_dist

_GRID_STEP = 0.1
_GRID_MAX = 4.0
_grid_vals = np.arange(0.1, _GRID_MAX + _GRID_STEP, _GRID_STEP)
_grid_n = len(_grid_vals)
_MAX_G = 8
_goals = np.arange(_MAX_G + 1)

# Precompute PMFs for all grid lambdas
_pmf_cache = np.array([_poisson_dist.pmf(_goals, lam) for lam in _grid_vals])  # shape: (n, 9)

# Precompute (p_home, p_draw, p_away) for every (lam_h, lam_a) combo — vectorized
# _pmf_cache shape: (n, G+1). Build all score matrices at once.
# all_mats[i,j,g1,g2] = P(home=g1|lam_i) * P(away=g2|lam_j)
_all_mats = np.einsum('ig,jh->ijgh', _pmf_cache, _pmf_cache)  # (n, n, G+1, G+1)

_g = _MAX_G + 1
_home_mask = np.tril(np.ones((_g, _g), dtype=bool), -1)  # g1 > g2
_draw_mask = np.eye(_g, dtype=bool)
_away_mask = np.triu(np.ones((_g, _g), dtype=bool), 1)   # g1 < g2

_grid_ph = np.sum(_all_mats * _home_mask, axis=(2, 3))
_grid_pd = np.sum(_all_mats * _draw_mask, axis=(2, 3))
_grid_pa = np.sum(_all_mats * _away_mask, axis=(2, 3))
del _all_mats  # free memory


def fit_poisson_lambdas(p_home: float, p_draw: float, p_away: float) -> tuple[float, float]:
    """
    Find Poisson lambdas via precomputed grid lookup.
    Finds the (lam_h, lam_a) pair whose implied probabilities best match target.
    """
    err = (_grid_ph - p_home)**2 + (_grid_pd - p_draw)**2 + (_grid_pa - p_away)**2
    idx = np.unravel_index(np.argmin(err), err.shape)
    return float(_grid_vals[idx[0]]), float(_grid_vals[idx[1]])


def most_likely_scoreline(lam_h: float, lam_a: float, max_goals: int = 6) -> tuple[int, int]:
    """
    Return the most probable scoreline given Poisson lambdas.
    Computes P(h, a) = Poisson(h|lam_h) * Poisson(a|lam_a) for all (h, a)
    and returns the pair with highest joint probability.
    """
    best_p, best_h, best_a = -1.0, 0, 0
    for h in range(max_goals + 1):
        ph = _poisson_dist.pmf(h, lam_h)
        for a in range(max_goals + 1):
            p = ph * _poisson_dist.pmf(a, lam_a)
            if p > best_p:
                best_p, best_h, best_a = p, h, a
    return best_h, best_a


# ── Feature builder ─────────────────────────────────────────

def build_features(home: str, away: str, neutral: bool = True) -> np.ndarray:
    """Build the 97-feature vector for a match."""
    home_elo = elos.get(home, 1500)
    away_elo = elos.get(away, 1500)
    elo_diff = home_elo - away_elo

    hf = get_form(home)
    af = get_form(away)
    h2h_hw, h2h_hs, h2h_hc, h2h_tm, h2h_rw = get_h2h(home, away)

    hc = conf_onehot(home)
    ac = conf_onehot(away)
    same_conf = int(confs.get(home, "X") == confs.get(away, "Y"))

    # Engineered form features
    home_form_momentum = hf["win_rate_5"] - hf["win_rate_10"]
    away_form_momentum = af["win_rate_5"] - af["win_rate_10"]
    home_gdf = hf["avg_scored_5"] - hf["avg_conceded_5"]
    away_gdf = af["avg_scored_5"] - af["avg_conceded_5"]
    net_goal_diff = home_gdf - away_gdf
    h2h_confidence = h2h_rw * (h2h_tm / (h2h_tm + 5))

    # Squad features
    hsq = get_squad_features(home)
    asq = get_squad_features(away)

    # 22 diffs
    sq_diffs = {f'{f}_diff': hsq[f] - asq[f] for f in SQUAD_FEATURES}

    # Engineered EA features
    h_overall = hsq['squad_avg_overall'] if not np.isnan(hsq.get('squad_avg_overall', np.nan)) else np.nan
    a_overall = asq['squad_avg_overall'] if not np.isnan(asq.get('squad_avg_overall', np.nan)) else np.nan
    h_top3 = hsq.get('squad_top3_avg', np.nan)
    a_top3 = asq.get('squad_top3_avg', np.nan)

    overall_ratio = h_overall / max(a_overall, 1) if not (np.isnan(h_overall) or np.isnan(a_overall)) else np.nan
    top3_ratio = h_top3 / max(a_top3, 1) if not (np.isnan(h_top3) or np.isnan(a_top3)) else np.nan

    h_val = hsq.get('squad_total_value', np.nan)
    a_val = asq.get('squad_total_value', np.nan)
    value_ratio_log = (np.log1p(h_val) - np.log1p(a_val)) if not (np.isnan(h_val) or np.isnan(a_val)) else np.nan
    value_ratio = ((h_val + 1) / (a_val + 1)) if not (np.isnan(h_val) or np.isnan(a_val)) else np.nan

    h_strong = hsq.get('strongest_unit', np.nan)
    h_weak = hsq.get('weakest_unit', np.nan)
    a_strong = asq.get('strongest_unit', np.nan)
    a_weak = asq.get('weakest_unit', np.nan)
    h_balance = h_strong - h_weak if not (np.isnan(h_strong) or np.isnan(h_weak)) else np.nan
    a_balance = a_strong - a_weak if not (np.isnan(a_strong) or np.isnan(a_weak)) else np.nan
    squad_balance_diff = h_balance - a_balance if not (np.isnan(h_balance) or np.isnan(a_balance)) else np.nan

    h_star_gap = h_top3 - h_overall if not (np.isnan(h_top3) or np.isnan(h_overall)) else np.nan
    a_star_gap = a_top3 - a_overall if not (np.isnan(a_top3) or np.isnan(a_overall)) else np.nan
    star_gap_diff = h_star_gap - a_star_gap if not (np.isnan(h_star_gap) or np.isnan(a_star_gap)) else np.nan

    depth_diff = sq_diffs.get('squad_bottom5_avg_diff', np.nan)
    squad_std_diff = sq_diffs.get('squad_std_overall_diff', np.nan)

    h_fwd = hsq.get('fwd_avg', np.nan)
    a_fwd = asq.get('fwd_avg', np.nan)
    h_def = hsq.get('def_avg', np.nan)
    a_def = asq.get('def_avg', np.nan)
    home_attack_vs_def = h_fwd - a_def if not (np.isnan(h_fwd) or np.isnan(a_def)) else np.nan
    away_attack_vs_def = a_fwd - h_def if not (np.isnan(a_fwd) or np.isnan(h_def)) else np.nan
    attack_vs_def_diff = home_attack_vs_def - away_attack_vs_def if not (np.isnan(home_attack_vs_def) or np.isnan(away_attack_vs_def)) else np.nan

    mid_battle = sq_diffs.get('mid_avg_diff', np.nan)
    gk_diff = sq_diffs.get('gk_avg_diff', np.nan)

    pace_diff = sq_diffs.get('team_pace_diff', np.nan)
    physic_diff = sq_diffs.get('team_physic_diff', np.nan)
    shooting_diff = sq_diffs.get('team_shooting_diff', np.nan)
    passing_diff = sq_diffs.get('team_passing_diff', np.nan)
    defending_diff = sq_diffs.get('team_defending_diff', np.nan)
    dribbling_diff = sq_diffs.get('team_dribbling_diff', np.nan)

    age_diff = sq_diffs.get('squad_avg_age_diff', np.nan)
    caps_diff = sq_diffs.get('squad_avg_caps_diff', np.nan)
    potential_gap_diff = sq_diffs.get('squad_avg_potential_gap_diff', np.nan)

    h_ws = 0.6 * h_overall + 0.4 * h_top3 if not (np.isnan(h_overall) or np.isnan(h_top3)) else np.nan
    a_ws = 0.6 * a_overall + 0.4 * a_top3 if not (np.isnan(a_overall) or np.isnan(a_top3)) else np.nan
    weighted_strength_diff = h_ws - a_ws if not (np.isnan(h_ws) or np.isnan(a_ws)) else np.nan

    elo_diff_sq = elo_diff ** 2 * np.sign(elo_diff)

    # Build feature dict matching FEATURE_COLS order
    feat = {
        # Form (25)
        'home_win_rate_5': hf['win_rate_5'], 'home_avg_scored_5': hf['avg_scored_5'],
        'home_avg_conceded_5': hf['avg_conceded_5'], 'home_pts_per_match_5': hf['pts_per_match_5'],
        'home_matches_played_5': hf['matches_played_5'],
        'home_win_rate_10': hf['win_rate_10'], 'home_avg_scored_10': hf['avg_scored_10'],
        'home_avg_conceded_10': hf['avg_conceded_10'], 'home_pts_per_match_10': hf['pts_per_match_10'],
        'home_matches_played_10': hf['matches_played_10'],
        'away_win_rate_5': af['win_rate_5'], 'away_avg_scored_5': af['avg_scored_5'],
        'away_avg_conceded_5': af['avg_conceded_5'], 'away_pts_per_match_5': af['pts_per_match_5'],
        'away_matches_played_5': af['matches_played_5'],
        'away_win_rate_10': af['win_rate_10'], 'away_avg_scored_10': af['avg_scored_10'],
        'away_avg_conceded_10': af['avg_conceded_10'], 'away_pts_per_match_10': af['pts_per_match_10'],
        'away_matches_played_10': af['matches_played_10'],
        'home_form_momentum': home_form_momentum, 'away_form_momentum': away_form_momentum,
        'home_goal_diff_form': home_gdf, 'away_goal_diff_form': away_gdf,
        'net_goal_diff': net_goal_diff,
        # H2H (6)
        'h2h_home_win_rate': h2h_hw, 'h2h_home_avg_scored': h2h_hs,
        'h2h_home_avg_conceded': h2h_hc, 'h2h_total_meetings': h2h_tm,
        'h2h_recent_win_rate': h2h_rw, 'h2h_confidence': h2h_confidence,
        # Context (17)
        'neutral.1': int(neutral), 'tournament_importance': 60,
        'home_conf_UEFA': hc['conf_UEFA'], 'home_conf_CAF': hc['conf_CAF'],
        'home_conf_AFC': hc['conf_AFC'], 'home_conf_CONCACAF': hc['conf_CONCACAF'],
        'home_conf_CONMEBOL': hc['conf_CONMEBOL'], 'home_conf_OFC': hc['conf_OFC'],
        'home_conf_UNKNOWN': hc['conf_UNKNOWN'],
        'away_conf_UEFA': ac['conf_UEFA'], 'away_conf_CAF': ac['conf_CAF'],
        'away_conf_AFC': ac['conf_AFC'], 'away_conf_CONCACAF': ac['conf_CONCACAF'],
        'away_conf_CONMEBOL': ac['conf_CONMEBOL'], 'away_conf_OFC': ac['conf_OFC'],
        'away_conf_UNKNOWN': ac['conf_UNKNOWN'],
        'same_confederation': same_conf,
        # EA diffs (22)
        **sq_diffs,
        # EA engineered (23)
        'overall_ratio': overall_ratio, 'top3_ratio': top3_ratio,
        'value_ratio_log': value_ratio_log, 'value_ratio': value_ratio,
        'squad_balance_diff': squad_balance_diff, 'star_gap_diff': star_gap_diff,
        'depth_diff': depth_diff, 'squad_std_diff': squad_std_diff,
        'home_attack_vs_def': home_attack_vs_def, 'away_attack_vs_def': away_attack_vs_def,
        'attack_vs_def_diff': attack_vs_def_diff, 'mid_battle': mid_battle, 'gk_diff': gk_diff,
        'pace_diff': pace_diff, 'physic_diff': physic_diff,
        'shooting_diff': shooting_diff, 'passing_diff': passing_diff,
        'defending_diff': defending_diff, 'dribbling_diff': dribbling_diff,
        'age_diff': age_diff, 'caps_diff': caps_diff,
        'potential_gap_diff': potential_gap_diff, 'weighted_strength_diff': weighted_strength_diff,
        # ELO (4)
        'elo_diff': elo_diff, 'elo_diff_sq': elo_diff_sq,
        'home_elo_before': home_elo, 'away_elo_before': away_elo,
    }

    return np.array([feat.get(c, np.nan) for c in FEATURE_COLS], dtype=float)


# ── Predict match ──────────────────────────────────────────

def predict_match(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """
    Returns (p_home_win, p_draw, p_away_win).
    XGB×3 + RF×1 blend (no DC).
    """
    feat = build_features(home, away, neutral).reshape(1, -1)

    # XGB handles NaN natively
    xgb_p = xgb_model.predict_proba(feat)[0]

    # RF needs NaN filled
    feat_rf = np.nan_to_num(feat, 0)
    rf_p = rf_model.predict_proba(feat_rf)[0]

    blended = (W_XGB * xgb_p + W_RF * rf_p) / (W_XGB + W_RF)

    # le.classes_ = ['away_win', 'draw', 'home_win']
    p_away = float(blended[0])
    p_draw = float(blended[1])
    p_home = float(blended[2])
    return p_home, p_draw, p_away


# ── Pre-cache all 48-team matchups ─────────────────────────

teams_2026 = pd.read_csv(DATA_DIR / "teams_2026.csv")
ALL_WC_TEAMS = teams_2026["team"].tolist()

prob_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
lambda_cache: dict[tuple[str, str], tuple[float, float]] = {}


def warm_cache():
    """Pre-compute all pairwise probabilities and Poisson lambdas for WC teams (batched)."""
    # Build all feature vectors first
    pairs = []
    feat_rows = []
    for home in ALL_WC_TEAMS:
        for away in ALL_WC_TEAMS:
            if home != away:
                pairs.append((home, away))
                feat_rows.append(build_features(home, away, neutral=True))

    X = np.array(feat_rows)  # (N, 97)
    X_rf = np.nan_to_num(X, 0)

    # Batch predict — much faster than one-at-a-time
    xgb_probs = xgb_model.predict_proba(X)    # (N, 3)
    rf_probs = rf_model.predict_proba(X_rf)    # (N, 3)
    blended = (W_XGB * xgb_probs + W_RF * rf_probs) / (W_XGB + W_RF)

    # le.classes_ = ['away_win', 'draw', 'home_win']
    for i, (home, away) in enumerate(pairs):
        p_away = float(blended[i, 0])
        p_draw = float(blended[i, 1])
        p_home = float(blended[i, 2])
        prob_cache[(home, away)] = (p_home, p_draw, p_away)
        lambda_cache[(home, away)] = fit_poisson_lambdas(p_home, p_draw, p_away)


def predict_match_cached(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """Use cache for WC teams, fall back to live prediction."""
    if (home, away) in prob_cache:
        return prob_cache[(home, away)]
    return predict_match(home, away, neutral)


def get_lambdas_cached(home: str, away: str) -> tuple[float, float]:
    """Get Poisson lambdas for scoreline generation."""
    if (home, away) in lambda_cache:
        return lambda_cache[(home, away)]
    ph, pd_, pa = predict_match(home, away, neutral=True)
    return fit_poisson_lambdas(ph, pd_, pa)
