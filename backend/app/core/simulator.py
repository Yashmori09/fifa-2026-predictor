"""
Tournament simulator — Phase 2.
Outcome decided by model probabilities first, then scoreline sampled to match.
This ensures match results follow model predictions faithfully while still
producing realistic, varied scorelines.
"""
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

def build_r32_bracket(winners: dict, runners_up: dict, best_thirds: dict) -> list[tuple[str, str]]:
    """
    Build Round of 32 bracket — 16 matchups, FIFA-style seeding:
      8 matches: group winners vs third-place qualifiers (cross-group)
      4 matches: remaining group winners vs cross-group runners-up
      4 matches: remaining runners-up vs runners-up
    Third-place teams face group winners (harder path), not each other.
    """
    all_groups = list("ABCDEFGHIJKL")
    third_groups = sorted(best_thirds.keys())
    non_third_groups = sorted(set(all_groups) - set(third_groups))

    # ── 8 matches: group winners vs thirds (cross-group) ──
    # Rotate third_groups by 4 to assign each third to a winner
    # from a different group (guaranteed no same-group collision)
    rotated_winner_groups = third_groups[4:] + third_groups[:4]
    wt_matches = []
    for i, tg in enumerate(third_groups):
        wg = rotated_winner_groups[i]
        wt_matches.append((winners[wg], best_thirds[tg]))

    # ── 4 matches: remaining winners vs cross-group runners-up ──
    # The 4 groups whose thirds didn't qualify — their winners face
    # runners-up from the other non-qualifying group (cross-paired)
    wr_matches = []
    for i in range(0, len(non_third_groups), 2):
        g1, g2 = non_third_groups[i], non_third_groups[i + 1]
        wr_matches.append((winners[g1], runners_up[g2]))
        wr_matches.append((winners[g2], runners_up[g1]))

    # ── 4 matches: remaining runners-up vs runners-up ──
    # Runners-up from the 8 third-qualifying groups, cross-paired
    rr_matches = []
    for i in range(0, len(third_groups), 2):
        g1, g2 = third_groups[i], third_groups[i + 1]
        rr_matches.append((runners_up[g1], runners_up[g2]))

    # Interleave for balanced bracket halves:
    # Left side (matches 0-7): first 4 winner-vs-third + 2 winner-vs-runner + 2 runner-vs-runner
    # Right side (matches 8-15): last 4 winner-vs-third + 2 winner-vs-runner + 2 runner-vs-runner
    left = wt_matches[:4] + wr_matches[:2] + rr_matches[:2]
    right = wt_matches[4:] + wr_matches[2:] + rr_matches[2:]

    return left + right


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
    Run one full tournament simulation.
    Returns complete bracket data for the frontend animation.
    """
    # Group stage
    group_result = simulate_all_groups()

    # Build R32 bracket
    r32_matchups = build_r32_bracket(
        group_result["winners"],
        group_result["runners_up"],
        group_result["best_thirds"],
    )

    # Simulate knockout rounds
    knockout = {}
    round_names = ["r32", "r16", "qf", "sf", "final"]

    current_teams = [t for match in r32_matchups for t in match]

    for round_idx, round_name in enumerate(round_names):
        round_matches = []
        next_teams = []
        for i in range(0, len(current_teams), 2):
            t1, t2 = current_teams[i], current_teams[i + 1]
            match = simulate_ko_match(t1, t2)
            round_matches.append(match)
            next_teams.append(match["winner"])
        knockout[round_name] = round_matches
        current_teams = next_teams
        if len(current_teams) == 1:
            break

    champion = current_teams[0]

    return {
        "champion": champion,
        "groups": group_result["groups"],
        "best_thirds": group_result["best_thirds"],
        "knockout": knockout,
    }
