"""
Match predictor — Phase 3 (hybrid goal-scoring).

Architecture: two XGB Poisson regressors predict (λ_h, λ_a); Dixon-Coles
correction applied to the joint scoreline matrix; W/D/L derived by summing
the matrix. Same external API as Phase 2 predictor for drop-in swap.

Features (~163):
  • Base: ELO before, form-5/10, H2H, confederations, DC-derived
  • Squad: FIFA + EA FC team ratings (per-year)
  • B5: StatsBomb international-tournament features, intl form, chemistry
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson as _poisson_dist

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

ALL_CONFS = ["UEFA", "CAF", "AFC", "CONCACAF", "CONMEBOL", "OFC", "UNKNOWN"]
MAX_GOALS = 10
_goals_grid = np.arange(MAX_GOALS + 1)

# ── Feature catalogs (must match training) ───────────────────────────────

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

# ── Load hybrid model + data ─────────────────────────────────────────────

_bundle = joblib.load(MODELS_DIR / "phase3_hybrid_clean.pkl")
MODEL_HOME = _bundle["model_home"]
MODEL_AWAY = _bundle["model_away"]
RHO = float(_bundle["rho"])
FEATURE_COLS = _bundle["features"]

elos = pd.read_csv(DATA_DIR / "final_elos.csv").set_index("team")["final_elo"].to_dict()
confs = pd.read_csv(DATA_DIR / "team_confederations.csv").set_index("team")["confederation"].to_dict()
fm = pd.read_csv(DATA_DIR / "features_matrix.csv", parse_dates=["date"]).sort_values("date")
team_features_by_year = pd.read_csv(DATA_DIR / "team_features_by_year.csv")

# Phase 3-new feature sources
sb_features = pd.read_csv(DATA_DIR / "team_statsbomb_features.csv").set_index("team")
intl_features = pd.read_csv(DATA_DIR / "team_intl_form_features.csv").set_index("team")
chem_features = pd.read_csv(DATA_DIR / "team_chemistry_features.csv").set_index("team")

# DC v2 (leakage-free)
dc_ratings = pd.read_csv(DATA_DIR / "dc_ratings_v2.csv")
DC_MAP = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}
with open(MODELS_DIR / "dc_params_v2.json") as f:
    DC_PARAMS = json.load(f)
DC_RHO = float(DC_PARAMS["rho"])

# Form lookups
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

_h2h_cols = [
    "h2h_home_win_rate", "h2h_home_avg_scored",
    "h2h_home_avg_conceded", "h2h_total_meetings", "h2h_recent_win_rate",
]
h2h_lookup = fm.groupby(["home_team", "away_team"])[_h2h_cols].last().to_dict("index")

# Pre-build squad lookup by (team, year)
_squad_lookup = {}
_tf_avail_years = sorted(team_features_by_year["year"].unique())
for _, _row in team_features_by_year.iterrows():
    _squad_lookup[(_row["team"], int(_row["year"]))] = {f: _row[f] for f in SQUAD_FEATURES}
_NAN_SQUAD = {f: np.nan for f in SQUAD_FEATURES}


def _squad_for(team: str, year: int = 2026) -> dict:
    cands = [y for y in _tf_avail_years if y <= year]
    if not cands:
        return _NAN_SQUAD
    return _squad_lookup.get((team, max(cands)), _NAN_SQUAD)


def _form_for(team: str) -> dict:
    if team in latest_home.index:
        r = latest_home.loc[team]
        return {
            "win_rate_5": r.get("home_win_rate_5", 0.5),
            "avg_scored_5": r.get("home_avg_scored_5", 1.3),
            "avg_conceded_5": r.get("home_avg_conceded_5", 1.0),
            "pts_per_match_5": r.get("home_pts_per_match_5", 1.5),
            "matches_played_5": r.get("home_matches_played_5", 5),
            "win_rate_10": r.get("home_win_rate_10", 0.5),
            "avg_scored_10": r.get("home_avg_scored_10", 1.3),
            "avg_conceded_10": r.get("home_avg_conceded_10", 1.0),
            "pts_per_match_10": r.get("home_pts_per_match_10", 1.5),
            "matches_played_10": r.get("home_matches_played_10", 10),
        }
    if team in latest_away.index:
        r = latest_away.loc[team]
        return {
            "win_rate_5": r.get("away_win_rate_5", 0.5),
            "avg_scored_5": r.get("away_avg_scored_5", 1.0),
            "avg_conceded_5": r.get("away_avg_conceded_5", 1.3),
            "pts_per_match_5": r.get("away_pts_per_match_5", 1.2),
            "matches_played_5": r.get("away_matches_played_5", 5),
            "win_rate_10": r.get("away_win_rate_10", 0.5),
            "avg_scored_10": r.get("away_avg_scored_10", 1.0),
            "avg_conceded_10": r.get("away_avg_conceded_10", 1.3),
            "pts_per_match_10": r.get("away_pts_per_match_10", 1.2),
            "matches_played_10": r.get("away_matches_played_10", 10),
        }
    return {
        "win_rate_5": 0.5, "avg_scored_5": 1.3, "avg_conceded_5": 1.3,
        "pts_per_match_5": 1.5, "matches_played_5": 5,
        "win_rate_10": 0.5, "avg_scored_10": 1.3, "avg_conceded_10": 1.3,
        "pts_per_match_10": 1.5, "matches_played_10": 10,
    }


def _h2h_for(home: str, away: str) -> tuple:
    if (home, away) in h2h_lookup:
        d = h2h_lookup[(home, away)]
        return (d["h2h_home_win_rate"], d["h2h_home_avg_scored"],
                d["h2h_home_avg_conceded"], d["h2h_total_meetings"],
                d["h2h_recent_win_rate"])
    return 0.5, 1.3, 1.3, 0, 0.5


def _dc_match(home: str, away: str, neutral: bool = True) -> dict:
    """Per-match DC-derived features (these feed the hybrid model)."""
    if home in DC_MAP and away in DC_MAP:
        h_att, h_def = DC_MAP[home]
        a_att, a_def = DC_MAP[away]
        ha = DC_PARAMS["home_adv"] if not neutral else 0.0
        lam = float(np.exp(h_att + a_def + ha))
        mu = float(np.exp(a_att + h_def))
    else:
        lam = mu = 1.4
    M = np.outer(_poisson_dist.pmf(_goals_grid, lam),
                  _poisson_dist.pmf(_goals_grid, mu))
    M[0, 0] *= max(1 - lam * mu * DC_RHO, 1e-10)
    M[0, 1] *= max(1 + lam * DC_RHO, 1e-10)
    M[1, 0] *= max(1 + mu * DC_RHO, 1e-10)
    M[1, 1] *= max(1 - DC_RHO, 1e-10)
    M = M / M.sum()
    hw = float(np.sum(M * (_goals_grid[:, None] > _goals_grid[None, :])))
    dr = float(np.sum(M * (_goals_grid[:, None] == _goals_grid[None, :])))
    aw = max(0.0, 1.0 - hw - dr)
    return {
        "dc_home_win_prob": hw, "dc_draw_prob": dr, "dc_away_win_prob": aw,
        "dc_lambda": lam, "dc_mu": mu,
        "dc_total_goals": lam + mu, "dc_goal_diff": lam - mu,
    }


def _safe_lookup(idx_df, team, col, default):
    if team in idx_df.index and col in idx_df.columns:
        v = idx_df.loc[team][col]
        return v if pd.notna(v) else default
    return default


def build_features(home: str, away: str, neutral: bool = True) -> np.ndarray:
    """Build the Phase 3 feature vector (length matches FEATURE_COLS)."""
    h_elo = elos.get(home, 1500)
    a_elo = elos.get(away, 1500)
    elo_diff = h_elo - a_elo

    hf = _form_for(home); af = _form_for(away)
    h2h = _h2h_for(home, away)
    dc = _dc_match(home, away, neutral=neutral)

    # Confederation flags
    h_conf = confs.get(home, "UNKNOWN")
    a_conf = confs.get(away, "UNKNOWN")
    home_conf = {f"home_conf_{c}": int(h_conf == c) for c in ALL_CONFS}
    away_conf = {f"away_conf_{c}": int(a_conf == c) for c in ALL_CONFS}

    # Squad
    hsq = _squad_for(home, 2026)
    asq = _squad_for(away, 2026)

    def _diff(f):
        h = hsq.get(f, np.nan); a = asq.get(f, np.nan)
        if pd.isna(h) or pd.isna(a):
            return np.nan
        return h - a

    sq_diffs = {
        "squad_avg_overall_diff": _diff("squad_avg_overall"),
        "squad_top3_avg_diff":    _diff("squad_top3_avg"),
        "squad_value_diff":       _diff("squad_total_value"),
        "def_avg_diff":           _diff("def_avg"),
        "mid_avg_diff":           _diff("mid_avg"),
        "fwd_avg_diff":           _diff("fwd_avg"),
        "team_shooting_diff":     _diff("team_shooting"),
        "team_passing_diff":      _diff("team_passing"),
        "team_defending_diff":    _diff("team_defending"),
    }

    # B5 features per side
    h_sb = {f"home_{k}": _safe_lookup(sb_features, home, k, np.nan) for k in SB_FEATURES}
    a_sb = {f"away_{k}": _safe_lookup(sb_features, away, k, np.nan) for k in SB_FEATURES}
    h_intl = {f"home_{k}": _safe_lookup(intl_features, home, k, np.nan) for k in INTL_FEATURES}
    a_intl = {f"away_{k}": _safe_lookup(intl_features, away, k, np.nan) for k in INTL_FEATURES}
    h_chem = {f"home_{k}": _safe_lookup(chem_features, home, k, np.nan) for k in CHEM_FEATURES}
    a_chem = {f"away_{k}": _safe_lookup(chem_features, away, k, np.nan) for k in CHEM_FEATURES}

    new_diffs = {
        "sb_xg_for_diff": h_sb["home_sb_xg_for_per_match"] - a_sb["away_sb_xg_for_per_match"],
        "sb_def_overperf_diff": h_sb["home_sb_def_overperformance"] - a_sb["away_sb_def_overperformance"],
        "sb_att_overperf_diff": h_sb["home_sb_att_overperformance"] - a_sb["away_sb_att_overperformance"],
        "intl_form_diff": h_intl["home_intl_form_last10"] - a_intl["away_intl_form_last10"],
        "intl_win_rate_diff": h_intl["home_intl_win_rate"] - a_intl["away_intl_win_rate"],
        "intl_gf_diff": h_intl["home_intl_goals_for_per_match"] - a_intl["away_intl_goals_for_per_match"],
        "intl_ga_diff": h_intl["home_intl_goals_against_per_match"] - a_intl["away_intl_goals_against_per_match"],
        "chem_same_club_diff": h_chem["home_same_club_top3_pct"] - a_chem["away_same_club_top3_pct"],
        "chem_caps_diff": h_chem["home_avg_intl_caps"] - a_chem["away_avg_intl_caps"],
    }

    home_form_momentum = hf["win_rate_5"] - hf["win_rate_10"]
    away_form_momentum = af["win_rate_5"] - af["win_rate_10"]
    home_gdf = hf["avg_scored_5"] - hf["avg_conceded_5"]
    away_gdf = af["avg_scored_5"] - af["avg_conceded_5"]
    net_gd = home_gdf - away_gdf
    h2h_conf = h2h[4] * (h2h[3] / (h2h[3] + 5))

    feat = {
        "home_win_rate_5": hf["win_rate_5"], "home_avg_scored_5": hf["avg_scored_5"],
        "home_avg_conceded_5": hf["avg_conceded_5"], "home_pts_per_match_5": hf["pts_per_match_5"],
        "home_matches_played_5": hf["matches_played_5"],
        "home_win_rate_10": hf["win_rate_10"], "home_avg_scored_10": hf["avg_scored_10"],
        "home_avg_conceded_10": hf["avg_conceded_10"], "home_pts_per_match_10": hf["pts_per_match_10"],
        "home_matches_played_10": hf["matches_played_10"],
        "away_win_rate_5": af["win_rate_5"], "away_avg_scored_5": af["avg_scored_5"],
        "away_avg_conceded_5": af["avg_conceded_5"], "away_pts_per_match_5": af["pts_per_match_5"],
        "away_matches_played_5": af["matches_played_5"],
        "away_win_rate_10": af["win_rate_10"], "away_avg_scored_10": af["avg_scored_10"],
        "away_avg_conceded_10": af["avg_conceded_10"], "away_pts_per_match_10": af["pts_per_match_10"],
        "away_matches_played_10": af["matches_played_10"],
        "h2h_home_win_rate": h2h[0], "h2h_home_avg_scored": h2h[1],
        "h2h_home_avg_conceded": h2h[2], "h2h_total_meetings": h2h[3],
        "h2h_recent_win_rate": h2h[4],
        "neutral.1": int(neutral), "tournament_importance": 60,
        **home_conf, **away_conf,
        "same_confederation": int(h_conf == a_conf),
        "elo_diff_sq": elo_diff ** 2 * np.sign(elo_diff),
        "home_form_momentum": home_form_momentum,
        "away_form_momentum": away_form_momentum,
        "home_goal_diff_form": home_gdf, "away_goal_diff_form": away_gdf,
        "net_goal_diff": net_gd, "h2h_confidence": h2h_conf,
        **dc,
        "home_elo_before": h_elo, "away_elo_before": a_elo, "elo_diff": elo_diff,
        **{f"home_{k}": hsq[k] for k in SQUAD_FEATURES},
        **{f"away_{k}": asq[k] for k in SQUAD_FEATURES},
        **sq_diffs,
        **h_sb, **a_sb, **h_intl, **a_intl, **h_chem, **a_chem,
        **new_diffs,
    }

    return np.array([feat.get(c, np.nan) for c in FEATURE_COLS], dtype=float)


# ── Scoreline derivation ──────────────────────────────────────────────────

def _scoreline_matrix(lam_h: float, lam_a: float) -> np.ndarray:
    M = np.outer(_poisson_dist.pmf(_goals_grid, lam_h),
                  _poisson_dist.pmf(_goals_grid, lam_a))
    M[0, 0] *= max(1 - lam_h * lam_a * RHO, 1e-10)
    M[0, 1] *= max(1 + lam_h * RHO, 1e-10)
    M[1, 0] *= max(1 + lam_a * RHO, 1e-10)
    M[1, 1] *= max(1 - RHO, 1e-10)
    return M / M.sum()


def _matrix_to_wdl(M: np.ndarray) -> tuple[float, float, float]:
    hw = float(np.sum(M * (_goals_grid[:, None] > _goals_grid[None, :])))
    dr = float(np.sum(M * (_goals_grid[:, None] == _goals_grid[None, :])))
    aw = max(0.0, 1.0 - hw - dr)
    return hw, dr, aw


def most_likely_scoreline(lam_h: float, lam_a: float, max_goals: int = 6) -> tuple[int, int]:
    """Return the expected scoreline = round(λ_h)-round(λ_a).

    Why expected (mean) instead of modal (argmax of joint Poisson):
      The Poisson distribution is concentrated near its mean but discrete.
      For a heavy favorite (e.g. λ_h=2.6, λ_a=0.7), the single most-likely
      scoreline is 2-0 (12.5% prob), but the mean is 2.6-0.7 which rounds
      to 3-1. The argmax systematically understates blowouts because the
      Poisson has a long right tail when λ is high. Showing round(λ_h, λ_a)
      gives a displayed scoreline that matches the model's actual goal
      expectations.
    """
    return int(round(lam_h)), int(round(lam_a))


# ── Predict + cache API (same as Phase 2 predictor) ──────────────────────

def predict_match(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    """Returns (p_home_win, p_draw, p_away_win)."""
    feat = build_features(home, away, neutral).reshape(1, -1)
    lam_h = float(MODEL_HOME.predict(feat)[0])
    lam_a = float(MODEL_AWAY.predict(feat)[0])
    return _matrix_to_wdl(_scoreline_matrix(lam_h, lam_a))


teams_2026 = pd.read_csv(DATA_DIR / "teams_2026.csv")
ALL_WC_TEAMS = teams_2026["team"].tolist()

prob_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
lambda_cache: dict[tuple[str, str], tuple[float, float]] = {}


def warm_cache() -> None:
    """Batched pre-compute for all 48-team pairings."""
    pairs = []
    feat_rows = []
    for home in ALL_WC_TEAMS:
        for away in ALL_WC_TEAMS:
            if home != away:
                pairs.append((home, away))
                feat_rows.append(build_features(home, away, neutral=True))
    X = np.array(feat_rows)
    lam_h_arr = MODEL_HOME.predict(X)
    lam_a_arr = MODEL_AWAY.predict(X)
    for i, (h, a) in enumerate(pairs):
        lh = float(lam_h_arr[i]); la = float(lam_a_arr[i])
        prob_cache[(h, a)] = _matrix_to_wdl(_scoreline_matrix(lh, la))
        lambda_cache[(h, a)] = (lh, la)


def predict_match_cached(home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
    if (home, away) in prob_cache:
        return prob_cache[(home, away)]
    return predict_match(home, away, neutral)


def get_lambdas_cached(home: str, away: str) -> tuple[float, float]:
    if (home, away) in lambda_cache:
        return lambda_cache[(home, away)]
    feat = build_features(home, away, neutral=True).reshape(1, -1)
    return float(MODEL_HOME.predict(feat)[0]), float(MODEL_AWAY.predict(feat)[0])


# Phase 2 callers might still reference this — no-op compat shim
def fit_poisson_lambdas(p_home: float, p_draw: float, p_away: float) -> tuple[float, float]:
    """Compat shim: returns approximate (λ_h, λ_a) from W/D/L probs."""
    lam_h = max(0.1, p_home * 2.5 + p_draw * 1.0)
    lam_a = max(0.1, p_away * 2.5 + p_draw * 1.0)
    return lam_h, lam_a
