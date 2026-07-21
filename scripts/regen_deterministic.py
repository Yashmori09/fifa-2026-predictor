"""
Generate the "The Prediction" deterministic bracket and write it as
frontend/src/lib/deterministic-data.ts.

Group stage: argmax-per-match (empirically best for group standings — see
exp_marginal_map_full.py results, argmax got 11/12 group winners vs 9/12 for
marginal MAP).

Knockouts: MARGINAL MAP with champion-first construction.
  - Champion = argmax(p_win) over all R32 teams (from 100K Monte Carlo sim)
  - For each slot on champion's path (R32, R16, QF, SF, F): force champion
  - For other slots: cascade — winner = argmax marginal probability among the
    2 already-committed winners feeding into that slot
  - Scoreline: expected score (round(λ_h)-round(λ_a)) nudged to match the
    chosen outcome

Rationale for the argmax→marginal-MAP switch on knockouts:
  Argmax-per-match locally picks per-pair winners without aggregating over the
  full tournament distribution. Under uncertainty, that can pick a champion
  who isn't the aggregate #1 (see: our old bracket picked France while home-
  page aggregate said Spain, matching reality). Marginal MAP picks the team
  most likely to occupy each slot — the standard method used by FiveThirtyEight
  SPI, Groll et al. WC papers, and the Kaplan-Garstka (2001) family. On WC
  2026, marginal MAP scored 69 pool points vs argmax's 39.
"""

import json
import sys
from pathlib import Path
from typing import Optional

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
    """Return the expected scoreline = round(λ_h)-round(λ_a), nudged to satisfy
    the chosen outcome.

    Why expected (mean) instead of modal (argmax of joint Poisson):
      The Poisson is concentrated near its mean but discrete. For a heavy
      favorite (λ_h=2.6, λ_a=0.7) the single most-likely scoreline is 2-0
      (12.5% prob), but the mean is 2.6-0.7 which rounds to 3-1. The argmax
      systematically understates blowouts because the Poisson has a long
      right tail when λ is high. Showing round(λ_h, λ_a) gives a scoreline
      that matches the model's actual goal expectations.

    If the rounded expected scoreline doesn't satisfy the chosen W/D/L
    outcome (e.g. λ=(1.3,1.1) rounds to 1-1 but outcome is "home"), nudge
    the appropriate side by 1.
    """
    hg = int(round(lam_h))
    ag = int(round(lam_a))

    if outcome == "home" and hg <= ag:
        hg = ag + 1
    elif outcome == "away" and ag <= hg:
        ag = hg + 1
    elif outcome == "draw" and hg != ag:
        # average the two and use it for both sides
        m = int(round((lam_h + lam_a) / 2))
        hg = ag = m

    return hg, ag


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


def det_ko_match(t1: str, t2: str, forced_winner: Optional[str] = None) -> dict:
    """Knockout match.

    If forced_winner is None: use argmax(p_home, p_draw, p_away) as before.
    If forced_winner is set (marginal MAP): use that team as winner and pick a
    consistent scoreline. Penalties=True only if the model's own argmax outcome
    would have been a draw (signals a very close match) — this matches how the
    UI treats penalties.
    """
    ph, pd_, pa = predict_match_cached(t1, t2, neutral=True)
    lam_h, lam_a = get_lambdas_cached(t1, t2)

    outs = {"home": ph, "draw": pd_, "away": pa}
    argmax_outcome = max(outs, key=outs.get)

    if forced_winner is None:
        outcome = argmax_outcome
        if outcome == "home":
            winner = t1; penalties = False
        elif outcome == "away":
            winner = t2; penalties = False
        else:
            winner = t1 if ph >= pa else t2
            outcome = "home" if winner == t1 else "away"
            penalties = True
    else:
        winner = forced_winner
        outcome = "home" if winner == t1 else "away"
        # Preserve penalties=True if the model's own top outcome was a draw
        penalties = (argmax_outcome == "draw")

    hg, ag = most_likely_score(lam_h, lam_a, outcome)

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

    # ── Knockouts (marginal MAP, champion-first) ──
    print("\nKnockouts (marginal MAP)...")

    sim_csv = ROOT / "data/processed/wc2026_simulation_phase3.csv"
    sim = pd.read_csv(sim_csv)
    marg = {r["team"]: {k: float(r[k]) for k in ["p_r16", "p_qf", "p_sf", "p_final", "p_win"]}
             for _, r in sim.iterrows()}

    def m_get(team, key):
        return marg.get(team, {}).get(key, 0.0)

    r32_bracket = build_r32_bracket(winners, runners_up, best_thirds)
    r32_match_ids = sorted(r32_bracket.keys())

    # R32_PAIRINGS uses match numbers (e.g. 73-88) as keys — keep as dict
    r32_pool_by_matchid = {mid: list(r32_bracket[mid]) for mid in r32_match_ids}
    # R16/QF/SF pools indexed by slot number (0-based)
    r16_pools = [list(set(r32_bracket[a]) | set(r32_bracket[b])) for a, b in R16_PAIRINGS]
    qf_pools = [list(set(r16_pools[a]) | set(r16_pools[b])) for a, b in QF_PAIRINGS]
    sf_pools = [list(set(qf_pools[a]) | set(qf_pools[b])) for a, b in SF_PAIRINGS]

    all_teams = list({t for pair in r32_pool_by_matchid.values() for t in pair})
    champion = max(all_teams, key=lambda t: m_get(t, "p_win"))
    print(f"  Champion (marginal MAP): {champion}  p_win={m_get(champion,'p_win'):.4f}")

    def slot_of_list(team, pools_list):
        return next(i for i, pool in enumerate(pools_list) if team in pool)

    def slot_of_dict(team, pool_dict):
        return next(mid for mid, pool in pool_dict.items() if team in pool)

    champ_slots = {
        "r32": slot_of_dict(champion, r32_pool_by_matchid),  # match ID
        "r16": slot_of_list(champion, r16_pools),
        "qf":  slot_of_list(champion, qf_pools),
        "sf":  slot_of_list(champion, sf_pools),
    }
    print(f"  Champion path: R32 match #{champ_slots['r32']} → R16 #{champ_slots['r16']} → QF #{champ_slots['qf']} → SF #{champ_slots['sf']} → Final")

    def pick_winner(t1, t2, marg_key, slot_i, champ_slot):
        """Cascade pick: champion is forced at their slot, else argmax marginal
        over the 2 committed feeder winners."""
        if slot_i == champ_slot:
            return champion
        return t1 if m_get(t1, marg_key) >= m_get(t2, marg_key) else t2

    # R32: for each pair, force champion if their pair, else argmax(p_r16)
    r32_matches = []
    r32_winners = {}
    for mn in r32_match_ids:
        t1, t2 = r32_bracket[mn]
        winner = pick_winner(t1, t2, "p_r16", mn, champ_slots["r32"])
        m = det_ko_match(t1, t2, forced_winner=winner)
        r32_matches.append(m)
        r32_winners[mn] = winner

    # R16: pair R32 winners per R16_PAIRINGS, pick winner via cascade
    r16_matches = []
    r16_winners = []
    for slot_i, (ma, mb) in enumerate(R16_PAIRINGS):
        t1, t2 = r32_winners[ma], r32_winners[mb]
        winner = pick_winner(t1, t2, "p_qf", slot_i, champ_slots["r16"])
        m = det_ko_match(t1, t2, forced_winner=winner)
        r16_matches.append(m)
        r16_winners.append(winner)

    # QF
    qf_matches = []
    qf_winners = []
    for slot_i, (ia, ib) in enumerate(QF_PAIRINGS):
        t1, t2 = r16_winners[ia], r16_winners[ib]
        winner = pick_winner(t1, t2, "p_sf", slot_i, champ_slots["qf"])
        m = det_ko_match(t1, t2, forced_winner=winner)
        qf_matches.append(m)
        qf_winners.append(winner)

    # SF
    sf_matches = []
    sf_winners = []
    for slot_i, (ia, ib) in enumerate(SF_PAIRINGS):
        t1, t2 = qf_winners[ia], qf_winners[ib]
        winner = pick_winner(t1, t2, "p_final", slot_i, champ_slots["sf"])
        m = det_ko_match(t1, t2, forced_winner=winner)
        sf_matches.append(m)
        sf_winners.append(winner)

    # Final: champion by construction
    final = det_ko_match(sf_winners[0], sf_winners[1], forced_winner=champion)
    print(f"  Final: {sf_winners[0]} vs {sf_winners[1]} → {champion}")

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
