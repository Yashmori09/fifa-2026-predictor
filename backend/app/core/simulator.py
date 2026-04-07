"""
Tournament simulator — replicates notebook 07 exactly.
Single simulation returning full bracket data for the frontend.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from app.core.predictor import (
    predict_match_cached,
    dc_match_probs,
    attack_params,
    defense_params,
    team_idx_dc,
    teams_2026,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"

schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv")

# ── Group stage ──────────────────────────────────────────────

def simulate_match_group(home: str, away: str) -> dict:
    """
    Simulate a group stage match.
    Returns dict with all match data for the frontend.
    """
    ph, pd_, pa = predict_match_cached(home, away, neutral=True)
    result = np.random.choice(["home", "draw", "away"], p=[ph, pd_, pa])

    # Sample scoreline from DC lambdas (neutral)
    hi = team_idx_dc.get(home, 0)
    ai = team_idx_dc.get(away, 0)
    dc_lam = np.exp(attack_params[hi] + defense_params[ai])  # neutral: no home_adv
    dc_mu = np.exp(attack_params[ai] + defense_params[hi])

    home_goals = int(np.random.poisson(dc_lam))
    away_goals = int(np.random.poisson(dc_mu))

    # Adjust scoreline to match the sampled result
    if result == "home" and home_goals <= away_goals:
        home_goals = away_goals + 1
    elif result == "away" and away_goals <= home_goals:
        away_goals = home_goals + 1
    elif result == "draw":
        away_goals = home_goals

    if result == "home":
        home_pts, away_pts = 3, 0
    elif result == "away":
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
    """Build Round of 32 bracket — 16 matchups."""
    W = winners
    R = runners_up
    thirds = list(best_thirds.values())

    return [
        # 12 winner vs runner-up (cross groups)
        (W["A"], R["B"]),
        (W["C"], R["D"]),
        (W["B"], R["A"]),
        (W["D"], R["C"]),
        (W["E"], R["F"]),
        (W["G"], R["H"]),
        (W["F"], R["E"]),
        (W["H"], R["G"]),
        (W["I"], R["J"]),
        (W["K"], R["L"]),
        (W["J"], R["I"]),
        (W["L"], R["K"]),
        # 4 best-thirds matchups
        (thirds[0], thirds[1]),
        (thirds[2], thirds[3]),
        (thirds[4], thirds[5]),
        (thirds[6], thirds[7]),
    ]


def simulate_ko_match(t1: str, t2: str) -> dict:
    """
    Simulate a knockout match. No draws — penalties if drawn.
    Returns match data for the frontend.
    """
    ph, pd_, pa = predict_match_cached(t1, t2, neutral=True)
    result = np.random.choice(["home", "draw", "away"], p=[ph, pd_, pa])

    penalties = False
    if result == "home":
        winner = t1
    elif result == "away":
        winner = t2
    else:
        # Penalties: renormalize home vs away
        p_home_et = ph / (ph + pa)
        winner = t1 if np.random.random() < p_home_et else t2
        penalties = True

    # Sample scoreline from DC
    hi = team_idx_dc.get(t1, 0)
    ai = team_idx_dc.get(t2, 0)
    dc_lam = np.exp(attack_params[hi] + defense_params[ai])
    dc_mu = np.exp(attack_params[ai] + defense_params[hi])

    t1_goals = int(np.random.poisson(dc_lam))
    t2_goals = int(np.random.poisson(dc_mu))

    if not penalties:
        if winner == t1 and t1_goals <= t2_goals:
            t1_goals = t2_goals + 1
        elif winner == t2 and t2_goals <= t1_goals:
            t2_goals = t1_goals + 1
    else:
        # Draw in regular time
        t2_goals = t1_goals

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
