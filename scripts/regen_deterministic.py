"""
Generate the "The Prediction" deterministic bracket and write it as
frontend/src/lib/deterministic-data.ts.

For every match:
  - Outcome = argmax(p_home, p_draw, p_away) — no randomness
  - Scoreline = most likely (hg, ag) under Poisson(lam_h)/Poisson(lam_a)
    consistent with the chosen outcome
  - KO draws → winner is the higher-prob side (h vs a, ignoring draw)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core.predictor import (
    warm_cache, prob_cache, lambda_cache, predict_match_cached, get_lambdas_cached,
)
from app.core.simulator import (
    teams_2026, schedule, build_r32_bracket,
    R16_PAIRINGS, QF_PAIRINGS, SF_PAIRINGS,
)


MAX_G = 8


def most_likely_score(lam_h: float, lam_a: float, outcome: str) -> tuple[int, int]:
    """Return (hg, ag) maximizing Poisson(lam_h)*Poisson(lam_a) subject to outcome."""
    pmf_h = poisson.pmf(np.arange(MAX_G + 1), lam_h)
    pmf_a = poisson.pmf(np.arange(MAX_G + 1), lam_a)
    grid = np.outer(pmf_h, pmf_a)  # grid[h, a]

    if outcome == "home":
        mask = np.tril(np.ones_like(grid, dtype=bool), -1)
    elif outcome == "away":
        mask = np.triu(np.ones_like(grid, dtype=bool), 1)
    else:
        mask = np.eye(MAX_G + 1, dtype=bool)

    masked = np.where(mask, grid, -1)
    idx = np.unravel_index(np.argmax(masked), masked.shape)
    return int(idx[0]), int(idx[1])


def det_group_match(home: str, away: str) -> dict:
    ph, pd_, pa = predict_match_cached(home, away, neutral=True)
    lam_h, lam_a = get_lambdas_cached(home, away)

    outs = {"home": ph, "draw": pd_, "away": pa}
    outcome = max(outs, key=outs.get)
    hg, ag = most_likely_score(lam_h, lam_a, outcome)

    if outcome == "home":
        h_pts, a_pts = 3, 0
    elif outcome == "away":
        h_pts, a_pts = 0, 3
    else:
        h_pts, a_pts = 1, 1

    return {
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag,
        "home_pts": h_pts, "away_pts": a_pts,
        "prob_home": round(ph, 4), "prob_draw": round(pd_, 4), "prob_away": round(pa, 4),
    }


def det_ko_match(t1: str, t2: str) -> dict:
    ph, pd_, pa = predict_match_cached(t1, t2, neutral=True)
    lam_h, lam_a = get_lambdas_cached(t1, t2)

    outs = {"home": ph, "draw": pd_, "away": pa}
    outcome = max(outs, key=outs.get)
    hg, ag = most_likely_score(lam_h, lam_a, outcome)
    penalties = False

    if outcome == "home":
        winner = t1
    elif outcome == "away":
        winner = t2
    else:
        # Draw → deterministic penalty: higher-prob side
        winner = t1 if ph >= pa else t2
        penalties = True

    return {
        "team1": t1, "team2": t2,
        "team1_goals": hg, "team2_goals": ag,
        "winner": winner, "penalties": penalties,
        "prob_team1": round(ph, 4), "prob_draw": round(pd_, 4), "prob_team2": round(pa, 4),
    }


def det_group(group_name: str, group_teams: list, group_matches: pd.DataFrame) -> dict:
    stats = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0, "ga": 0, "mp": 0,
                 "w": 0, "d": 0, "l": 0} for t in group_teams}
    matches = []

    for _, row in group_matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        m = det_group_match(h, a)
        matches.append(m)
        stats[h]["pts"] += m["home_pts"]; stats[a]["pts"] += m["away_pts"]
        stats[h]["gf"] += m["home_goals"]; stats[h]["ga"] += m["away_goals"]
        stats[a]["gf"] += m["away_goals"]; stats[a]["ga"] += m["home_goals"]
        stats[h]["gd"] += m["home_goals"] - m["away_goals"]
        stats[a]["gd"] += m["away_goals"] - m["home_goals"]
        stats[h]["mp"] += 1; stats[a]["mp"] += 1
        if m["home_pts"] == 3: stats[h]["w"] += 1; stats[a]["l"] += 1
        elif m["away_pts"] == 3: stats[a]["w"] += 1; stats[h]["l"] += 1
        else: stats[h]["d"] += 1; stats[a]["d"] += 1

    table = sorted(stats.values(), key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return {"group": group_name, "matches": matches, "table": table}


def main():
    print("Warming cache...")
    warm_cache()

    # ── Group stage ──
    print("Group stage...")
    groups_data = []
    winners = {}
    runners_up = {}
    thirds = []
    for gname in sorted(teams_2026["group"].unique()):
        g_teams = teams_2026[teams_2026["group"] == gname]["team"].tolist()
        g_match = schedule[schedule["group"] == gname]
        r = det_group(gname, g_teams, g_match)
        groups_data.append(r)
        winners[gname] = r["table"][0]["team"]
        runners_up[gname] = r["table"][1]["team"]
        thirds.append({"team": r["table"][2]["team"], "pts": r["table"][2]["pts"],
                       "gd": r["table"][2]["gd"], "gf": r["table"][2]["gf"], "group": gname})

    thirds_sorted = sorted(thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    best_thirds = {d["group"]: d["team"] for d in thirds_sorted[:8]}
    print(f"  Winners: {winners}")
    print(f"  Best thirds: {best_thirds}")

    # ── R32 bracket ──
    print("\nKnockouts...")
    r32_bracket = build_r32_bracket(winners, runners_up, best_thirds)
    r32_matches = []
    r32_winners = {}
    for mn in sorted(r32_bracket.keys()):
        t1, t2 = r32_bracket[mn]
        m = det_ko_match(t1, t2)
        r32_matches.append(m)
        r32_winners[mn] = m["winner"]

    # ── R16 ──
    r16_matches = []
    r16_winners = []
    for ma, mb in R16_PAIRINGS:
        m = det_ko_match(r32_winners[ma], r32_winners[mb])
        r16_matches.append(m)
        r16_winners.append(m["winner"])

    # ── QF ──
    qf_matches = []
    qf_winners = []
    for ia, ib in QF_PAIRINGS:
        m = det_ko_match(r16_winners[ia], r16_winners[ib])
        qf_matches.append(m)
        qf_winners.append(m["winner"])

    # ── SF ──
    sf_matches = []
    sf_winners = []
    for ia, ib in SF_PAIRINGS:
        m = det_ko_match(qf_winners[ia], qf_winners[ib])
        sf_matches.append(m)
        sf_winners.append(m["winner"])

    # ── Final ──
    final = det_ko_match(sf_winners[0], sf_winners[1])
    champion = final["winner"]
    print(f"  Champion: {champion}")

    # ── Assemble ──
    result = {
        "champion": champion,
        "groups": groups_data,
        "best_thirds": best_thirds,
        "knockout": {
            "r32": r32_matches,
            "r16": r16_matches,
            "qf": qf_matches,
            "sf": sf_matches,
            "final": [final],
        },
    }

    # ── Write TS file ──
    ts_out = ROOT / "frontend/src/lib/deterministic-data.ts"
    ts_header = '''/**
 * Pre-computed deterministic prediction results.
 * Generated by scripts/regen_deterministic.py
 *
 * This is the model's single best answer — argmax of predicted probabilities
 * for every match. Unlike Monte Carlo simulation, this always produces the
 * same result.
 */

export const DETERMINISTIC_DATA = '''
    body = json.dumps(result, ensure_ascii=False, indent=2)
    ts_out.write_text(ts_header + body + ";\n", encoding="utf-8")
    print(f"\nWrote {ts_out} ({ts_out.stat().st_size:,} bytes)")
    print(f"Champion: {champion}")


if __name__ == "__main__":
    main()
