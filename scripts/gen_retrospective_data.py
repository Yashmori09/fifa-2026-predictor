"""
Generate per-team WC 2026 retrospective data for the /my-team page.

For each of 48 teams, computes:
  - Pre-tournament forecast (from 100K sim marginals)
  - Actual finish stage (from live API)
  - Elimination match + model's pre-match prediction of it
  - Aggregate stats (matches played, W/D/L, GF/GA)
  - Verdict (overachiever / matched / underperformed)

Output: frontend/public/retrospective.json
"""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIM_CSV = ROOT / "data/processed/wc2026_simulation_phase3.csv"
OUT_JSON = ROOT / "frontend/public/retrospective.json"
LIVE_URL = "https://fifa-2026-predictor.vercel.app/api/live-matches"

STAGE_ORDER = ["group", "r32", "r16", "qf", "sf", "tp", "final"]  # tp = third place playoff
STAGE_LABEL_REACHED = {
    "group": "Group stage",
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarterfinal",
    "sf": "Semifinal (4th place)",
    "tp": "3rd place",
    "final": "Final (runner-up)",
    "champion": "Champion",
}
# What round should we consider a team "expected to reach" based on marginals?
# We say a team is expected to reach a stage if their marginal probability is >= 0.5.
STAGE_TO_MARG = {"r32": "p_r16", "r16": "p_qf", "qf": "p_sf", "sf": "p_final", "final": "p_win"}


def load_marginals():
    sim = pd.read_csv(SIM_CSV)
    marg = {}
    for _, r in sim.iterrows():
        marg[r["team"]] = {
            "p_r16": float(r["p_r16"]),
            "p_qf": float(r["p_qf"]),
            "p_sf": float(r["p_sf"]),
            "p_final": float(r["p_final"]),
            "p_win": float(r["p_win"]),
        }
    return marg


def load_live():
    print(f"Fetching {LIVE_URL} ...")
    req = urllib.request.Request(LIVE_URL)
    data = json.loads(urllib.request.urlopen(req).read())
    return data


def expected_finish(marg):
    """Furthest stage team is expected to reach — the highest stage where their
    marginal reach-probability >= 0.5. p_r16 = P(make it out of groups), etc."""
    if marg.get("p_win", 0) >= 0.5: return "Champion"
    if marg.get("p_final", 0) >= 0.5: return "Final"
    if marg.get("p_sf", 0) >= 0.5: return "Semifinal"
    if marg.get("p_qf", 0) >= 0.5: return "Quarterfinal"
    if marg.get("p_r16", 0) >= 0.5: return "Round of 16"
    return "Group stage"


def compute_team_data(team_name, team_matches, marg_data, all_teams_meta):
    """Given all matches this team played, compute their retrospective."""
    finished_matches = [m for m in team_matches if m["status"] == "FINISHED"]
    if not finished_matches:
        return None

    # Determine actual finish
    stages_reached = [m["stage"] for m in finished_matches]
    max_stage = max(stages_reached, key=lambda s: STAGE_ORDER.index(s))

    def match_winner(m):
        if not m.get("actual") or m["actual"]["outcome"] == "draw":
            return None
        return m["home"]["name"] if m["actual"]["outcome"] == "home" else m["away"]["name"]

    is_champion = any(m["stage"] == "final" and match_winner(m) == team_name
                       for m in finished_matches)

    # Team's absolute finish (accounts for 3rd-place playoff and Final)
    if is_champion:
        actual_finish_stage = "champion"
    elif max_stage == "final":
        actual_finish_stage = "final"  # runner-up
    elif max_stage == "tp":
        tp = next(m for m in finished_matches if m["stage"] == "tp")
        actual_finish_stage = "tp" if match_winner(tp) == team_name else "sf"  # 3rd or 4th
    else:
        actual_finish_stage = max_stage

    # Aggregate stats
    mp = w = d_ = l = gf = ga = 0
    for m in finished_matches:
        act = m.get("actual")
        if not act:
            continue
        mp += 1
        is_home = m["home"]["name"] == team_name
        team_g = act["score"]["home"] if is_home else act["score"]["away"]
        opp_g = act["score"]["away"] if is_home else act["score"]["home"]
        gf += team_g
        ga += opp_g
        outcome = act["outcome"]
        if outcome == "draw":
            d_ += 1
        elif (outcome == "home" and is_home) or (outcome == "away" and not is_home):
            w += 1
        else:
            l += 1

    # Elimination match: last knockout match they played that they LOST
    # (or the Final if they made it but didn't win)
    elim_match = None
    if not is_champion:
        knockout_lost = [
            m for m in finished_matches
            if m["stage"] in ["r32", "r16", "qf", "sf", "final"] and m.get("actual")
            and m["actual"]["outcome"] != "draw"
            and (
                (m["home"]["name"] == team_name and m["actual"]["outcome"] == "away")
                or (m["away"]["name"] == team_name and m["actual"]["outcome"] == "home")
            )
        ]
        # Take the latest knockout loss by stage
        if knockout_lost:
            elim_match = max(knockout_lost, key=lambda m: STAGE_ORDER.index(m["stage"]))
        else:
            # Never made it out of groups — pick the last group match they LOST or DREW
            group_losses = [m for m in finished_matches if m["stage"] == "group"]
            if group_losses:
                elim_match = group_losses[-1]

    elim_data = None
    if elim_match:
        act = elim_match["actual"]
        pred = elim_match.get("prediction")
        is_home = elim_match["home"]["name"] == team_name
        opp_name = elim_match["away"]["name"] if is_home else elim_match["home"]["name"]
        opp_flag = elim_match["away"]["flag"] if is_home else elim_match["home"]["flag"]
        team_g = act["score"]["home"] if is_home else act["score"]["away"]
        opp_g = act["score"]["away"] if is_home else act["score"]["home"]

        elim_data = {
            "opponent": opp_name,
            "opponent_flag": opp_flag,
            "score": f"{team_g}-{opp_g}",
            "stage": elim_match["stage"],
            "stage_label": STAGE_LABEL_REACHED.get(elim_match["stage"], elim_match["stage"]),
        }
        if pred:
            # From this team's perspective: our win prob is prob_home if we're home
            our_win_prob = pred["prob_home"] if is_home else pred["prob_away"]
            opp_win_prob = pred["prob_away"] if is_home else pred["prob_home"]
            elim_data["our_win_prob"] = round(our_win_prob, 4)
            elim_data["opp_win_prob"] = round(opp_win_prob, 4)
            elim_data["draw_prob"] = round(pred["prob_draw"], 4)
            # Our pre-match predicted outcome (from this team's view)
            preds = {"win": our_win_prob, "draw": pred["prob_draw"], "loss": opp_win_prob}
            elim_data["our_pick"] = max(preds, key=preds.get)

    marg = marg_data.get(team_name, {})
    if not marg:
        # Fallback for teams not in sim
        marg = {"p_r16": 0, "p_qf": 0, "p_sf": 0, "p_final": 0, "p_win": 0}

    # Verdict: compare actual finish rank to expected
    stage_rank = {"group": 0, "r32": 1, "r16": 2, "qf": 3, "sf": 4, "tp": 5, "final": 6, "champion": 7}
    actual_rank = stage_rank[actual_finish_stage]
    exp_finish_str = expected_finish(marg)
    exp_rank = {
        "Group stage": 0, "Round of 32": 1, "Round of 16": 2,
        "Quarterfinal": 3, "Semifinal": 4, "Final": 6, "Champion": 7,
    }[exp_finish_str]
    if actual_rank > exp_rank:
        verdict = "overachieved"
    elif actual_rank == exp_rank:
        verdict = "matched"
    else:
        verdict = "underperformed"

    return {
        "team": team_name,
        "flag": (team_matches[0]["home"]["flag"] if team_matches[0]["home"]["name"] == team_name
                  else team_matches[0]["away"]["flag"]),
        "predicted": {
            "p_r16": round(marg["p_r16"], 4),
            "p_qf": round(marg["p_qf"], 4),
            "p_sf": round(marg["p_sf"], 4),
            "p_final": round(marg["p_final"], 4),
            "p_win": round(marg["p_win"], 4),
            "expected_finish": exp_finish_str,
        },
        "actual": {
            "finish_stage": actual_finish_stage,
            "finish_label": STAGE_LABEL_REACHED[actual_finish_stage],
            "matches_played": mp,
            "wins": w,
            "draws": d_,
            "losses": l,
            "gf": gf,
            "ga": ga,
        },
        "elimination": elim_data,
        "verdict": verdict,
    }


def main():
    print("Loading marginals...")
    marg = load_marginals()

    print("Loading live tournament data...")
    live = load_live()
    matches = live["matches"]

    # Group all matches by team name
    all_team_names = set()
    for m in matches:
        if m["home"]["name"]:
            all_team_names.add(m["home"]["name"])
        if m["away"]["name"]:
            all_team_names.add(m["away"]["name"])

    # Include only real WC 2026 teams (filter out TBD, None, etc.)
    all_team_names = {t for t in all_team_names if t and t not in ["TBD"]}

    team_data = {}
    for team in sorted(all_team_names):
        team_matches = [m for m in matches
                        if m["home"]["name"] == team or m["away"]["name"] == team]
        if not team_matches:
            continue
        result = compute_team_data(team, team_matches, marg, all_team_names)
        if result:
            team_data[team] = result

    # Overall stats
    overall = {
        "n_teams": len(team_data),
        "champion": next((t["team"] for t in team_data.values()
                          if t["actual"]["finish_stage"] == "champion"), None),
        "runner_up": next((t["team"] for t in team_data.values()
                            if t["actual"]["finish_stage"] == "final"), None),
        "stats": {
            "n_played": live["stats"]["n_played"],
            "outcome_accuracy": live["stats"].get("outcome_accuracy"),
            "avg_confidence_score": live["stats"].get("avg_confidence_score"),
            "n_upsets": live["stats"].get("n_upsets"),
        },
    }

    payload = {
        "generated_at": live["last_updated"],
        "overall": overall,
        "teams": team_data,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUT_JSON}")
    print(f"  {len(team_data)} teams")
    print(f"  Champion: {overall['champion']}")
    print(f"  Runner-up: {overall['runner_up']}")

    # Preview
    if overall["champion"]:
        print(f"\nSample — {overall['champion']}:")
        t = team_data[overall["champion"]]
        print(f"  Predicted p_win: {t['predicted']['p_win']}")
        print(f"  Verdict: {t['verdict']}")


if __name__ == "__main__":
    main()
