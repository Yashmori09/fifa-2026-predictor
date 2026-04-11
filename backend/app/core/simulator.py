"""
Tournament simulator — Phase 2.
Outcome decided by model probabilities first, then scoreline sampled to match.
This ensures match results follow model predictions faithfully while still
producing realistic, varied scorelines.

Bracket: Official FIFA 2026 R32 (Matches 73-88) with Annex C third-place
allocation (495 combinations) and fixed R16/QF/SF pairings.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from app.core.predictor import (
    predict_match_cached,
    get_lambdas_cached,
    teams_2026,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"

schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv")

# ── Official FIFA 2026 third-place allocation (Annex C, 495 combinations) ──
with open(DATA_DIR / "third_place_allocation.json") as _f:
    _THIRD_PLACE_TABLE = json.load(_f)
_THIRD_ALLOC_INDEX = {
    frozenset(entry["qualifying_groups"]): entry["assignments"]
    for entry in _THIRD_PLACE_TABLE
}

# ── Official FIFA knockout bracket path ──
R16_PAIRINGS = [(74, 77), (73, 75), (76, 78), (79, 80), (83, 84), (81, 82), (86, 88), (85, 87)]
QF_PAIRINGS  = [(0, 1), (4, 5), (2, 3), (6, 7)]   # indices into R16 winners
SF_PAIRINGS  = [(0, 1), (2, 3)]                      # indices into QF winners


# ── Scoreline sampling ──────────────────────────────────────

def _sample_scoreline(lam_h: float, lam_a: float, outcome: str) -> tuple[int, int]:
    """Sample a Poisson scoreline consistent with the chosen outcome."""
    for _ in range(100):
        hg = int(np.random.poisson(lam_h))
        ag = int(np.random.poisson(lam_a))
        if outcome == "home" and hg > ag:
            return hg, ag
        if outcome == "away" and ag > hg:
            return hg, ag
        if outcome == "draw" and hg == ag:
            return hg, ag
    # Fallback if rejection sampling exhausts
    if outcome == "home":
        return max(1, int(np.random.poisson(lam_h))), 0
    elif outcome == "away":
        return 0, max(1, int(np.random.poisson(lam_a)))
    else:
        g = int(np.random.poisson(min(lam_h, lam_a)))
        return g, g


# ── Group stage ──────────────────────────────────────────────

def simulate_match_group(home: str, away: str) -> dict:
    """
    Simulate a group stage match.
    Outcome picked from model probabilities, then scoreline sampled to match.
    """
    ph, pd_, pa = predict_match_cached(home, away, neutral=True)
    lam_h, lam_a = get_lambdas_cached(home, away)

    # Pick outcome using model probabilities directly
    outcome = np.random.choice(["home", "draw", "away"], p=[ph, pd_, pa])
    home_goals, away_goals = _sample_scoreline(lam_h, lam_a, outcome)

    if outcome == "home":
        home_pts, away_pts = 3, 0
    elif outcome == "away":
        home_pts, away_pts = 0, 3
    else:
        home_pts, away_pts = 1, 1

    return {
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "prob_home": round(ph, 4),
        "prob_draw": round(pd_, 4),
        "prob_away": round(pa, 4),
    }


def simulate_group(group_name: str, group_teams: list[str], group_matches: pd.DataFrame) -> dict:
    """
    Simulate one group. Returns standings + match results.
    """
    stats = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0, "ga": 0, "mp": 0, "w": 0, "d": 0, "l": 0}
             for t in group_teams}
    matches = []

    for _, row in group_matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        m = simulate_match_group(h, a)
        matches.append(m)

        stats[h]["pts"] += m["home_pts"]
        stats[a]["pts"] += m["away_pts"]
        stats[h]["gf"] += m["home_goals"]
        stats[h]["ga"] += m["away_goals"]
        stats[a]["gf"] += m["away_goals"]
        stats[a]["ga"] += m["home_goals"]
        stats[h]["gd"] += m["home_goals"] - m["away_goals"]
        stats[a]["gd"] += m["away_goals"] - m["home_goals"]
        stats[h]["mp"] += 1
        stats[a]["mp"] += 1

        if m["home_pts"] == 3:
            stats[h]["w"] += 1
            stats[a]["l"] += 1
        elif m["away_pts"] == 3:
            stats[a]["w"] += 1
            stats[h]["l"] += 1
        else:
            stats[h]["d"] += 1
            stats[a]["d"] += 1

    standing = sorted(stats.values(), key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)

    return {
        "group": group_name,
        "matches": matches,
        "table": standing,
    }


def simulate_all_groups() -> dict:
    """Simulate all 12 groups. Returns full group data + qualified teams."""
    groups_data = []
    winners = {}
    runners_up = {}
    third_place = []

    for group_name in sorted(teams_2026["group"].unique()):
        group_teams = teams_2026[teams_2026["group"] == group_name]["team"].tolist()
        group_matches = schedule[schedule["group"] == group_name]
        result = simulate_group(group_name, group_teams, group_matches)
        groups_data.append(result)

        winners[group_name] = result["table"][0]["team"]
        runners_up[group_name] = result["table"][1]["team"]
        third = result["table"][2]
        third_place.append({
            "team": third["team"],
            "pts": third["pts"],
            "gd": third["gd"],
            "gf": third["gf"],
            "group": group_name,
        })

    # Best 8 third-place teams
    third_sorted = sorted(third_place, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    best_thirds = {d["group"]: d["team"] for d in third_sorted[:8]}

    return {
        "groups": groups_data,
        "winners": winners,
        "runners_up": runners_up,
        "best_thirds": best_thirds,
        "third_place_sorted": third_sorted,
    }


# ── Knockout ─────────────────────────────────────────────────

def build_r32_bracket(winners: dict, runners_up: dict, best_thirds: dict) -> dict[int, tuple[str, str]]:
    """
    Official FIFA 2026 R32 bracket (Matches 73-88).
    Source: FIFA match schedule + Annex C third-place allocation.

    Fixed matches (no third-place dependency):
      M73: 2A vs 2B    M75: 1F vs 2C    M76: 1C vs 2F    M78: 2E vs 2I
      M83: 2K vs 2L    M84: 1H vs 2J    M86: 1J vs 2H    M88: 2D vs 2G

    Third-place matches (allocation from Annex C):
      M74: 1E vs 3rd    M77: 1I vs 3rd    M79: 1A vs 3rd    M80: 1L vs 3rd
      M81: 1D vs 3rd    M82: 1G vs 3rd    M85: 1B vs 3rd    M87: 1K vs 3rd
    """
    key = frozenset(best_thirds.keys())
    alloc = _THIRD_ALLOC_INDEX[key]

    r32 = {}
    # Fixed matches
    r32[73] = (runners_up["A"], runners_up["B"])
    r32[75] = (winners["F"], runners_up["C"])
    r32[76] = (winners["C"], runners_up["F"])
    r32[78] = (runners_up["E"], runners_up["I"])
    r32[83] = (runners_up["K"], runners_up["L"])
    r32[84] = (winners["H"], runners_up["J"])
    r32[86] = (winners["J"], runners_up["H"])
    r32[88] = (runners_up["D"], runners_up["G"])

    # Third-place matches
    r32[74] = (winners["E"], best_thirds[alloc["74"]])
    r32[77] = (winners["I"], best_thirds[alloc["77"]])
    r32[79] = (winners["A"], best_thirds[alloc["79"]])
    r32[80] = (winners["L"], best_thirds[alloc["80"]])
    r32[81] = (winners["D"], best_thirds[alloc["81"]])
    r32[82] = (winners["G"], best_thirds[alloc["82"]])
    r32[85] = (winners["B"], best_thirds[alloc["85"]])
    r32[87] = (winners["K"], best_thirds[alloc["87"]])

    return r32


def simulate_ko_match(t1: str, t2: str) -> dict:
    """
    Simulate a knockout match.
    Outcome picked from model probabilities. Draw → penalties decided by
    normalized home/away probs.
    """
    ph, pd_, pa = predict_match_cached(t1, t2, neutral=True)
    lam_h, lam_a = get_lambdas_cached(t1, t2)

    # Pick outcome using model probabilities directly
    outcome = np.random.choice(["home", "draw", "away"], p=[ph, pd_, pa])
    t1_goals, t2_goals = _sample_scoreline(lam_h, lam_a, outcome)

    penalties = False
    if outcome == "home":
        winner = t1
    elif outcome == "away":
        winner = t2
    else:
        # Draw → penalties: use model probabilities to decide
        p_t1_pen = ph / (ph + pa)
        winner = t1 if np.random.random() < p_t1_pen else t2
        penalties = True

    return {
        "team1": t1,
        "team2": t2,
        "team1_goals": t1_goals,
        "team2_goals": t2_goals,
        "winner": winner,
        "penalties": penalties,
        "prob_team1": round(ph, 4),
        "prob_draw": round(pd_, 4),
        "prob_team2": round(pa, 4),
    }


# ── Full tournament ──────────────────────────────────────────

def simulate_tournament() -> dict:
    """
    Run one full tournament simulation using official FIFA 2026 bracket.
    Returns complete bracket data for the frontend animation.
    """
    # Group stage
    group_result = simulate_all_groups()

    # Build R32 bracket (dict keyed by match number 73-88)
    r32 = build_r32_bracket(
        group_result["winners"],
        group_result["runners_up"],
        group_result["best_thirds"],
    )

    # ── R32: play matches 73-88 ──
    r32_matches = []
    r32_winners = {}
    for match_num in sorted(r32.keys()):
        t1, t2 = r32[match_num]
        match = simulate_ko_match(t1, t2)
        r32_matches.append(match)
        r32_winners[match_num] = match["winner"]

    # ── R16: official FIFA pairings ──
    r16_matches = []
    r16_winners = []
    for m_a, m_b in R16_PAIRINGS:
        t1, t2 = r32_winners[m_a], r32_winners[m_b]
        match = simulate_ko_match(t1, t2)
        r16_matches.append(match)
        r16_winners.append(match["winner"])

    # ── QF: indices into r16_winners ──
    qf_matches = []
    qf_winners = []
    for i_a, i_b in QF_PAIRINGS:
        t1, t2 = r16_winners[i_a], r16_winners[i_b]
        match = simulate_ko_match(t1, t2)
        qf_matches.append(match)
        qf_winners.append(match["winner"])

    # ── SF ──
    sf_matches = []
    sf_winners = []
    for i_a, i_b in SF_PAIRINGS:
        t1, t2 = qf_winners[i_a], qf_winners[i_b]
        match = simulate_ko_match(t1, t2)
        sf_matches.append(match)
        sf_winners.append(match["winner"])

    # ── Final ──
    final_match = simulate_ko_match(sf_winners[0], sf_winners[1])
    champion = final_match["winner"]

    return {
        "champion": champion,
        "groups": group_result["groups"],
        "best_thirds": group_result["best_thirds"],
        "knockout": {
            "r32": r32_matches,
            "r16": r16_matches,
            "qf": qf_matches,
            "sf": sf_matches,
            "final": [final_match],
        },
    }
