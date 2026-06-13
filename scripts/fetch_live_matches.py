"""
Fetch live and upcoming FIFA 2026 WC matches from football-data.org,
join with our pre-tournament predictions (from deterministic-data.ts),
and write frontend/src/data/live_matches.json.

For each match:
  - Group stage matches: predictions come from the frozen deterministic bracket
    (computed pre-WC and shipped with the frontend).
  - KO matches: prediction added when both teams are confirmed via a separate
    pipeline (predict_pending_matches.py — not built yet, since groups run first).

Outputs aggregate accuracy stats: outcome correct %, exact-score %, within-1-goal %,
upset detection (favorite >0.6 prob lost).

Usage:
  export FOOTBALL_DATA_API_KEY=your_token
  python scripts/fetch_live_matches.py
"""

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE_OUT = ROOT / "frontend/src/data/live_matches.json"
DET_PATH = ROOT / "frontend/src/lib/deterministic-data.ts"

API_BASE = "https://api.football-data.org/v4"
COMP_CODE = "WC"
SEASON = 2026

# football-data.org uses different team names than ours. Map their names → ours.
TEAM_NAME_MAP = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "USA": "United States",
    # Add others as they surface
}

# Country code (TLA) → our flag-icons ISO 3166-1 alpha-2 code for the frontend
TLA_TO_FLAG = {
    "ALG": "dz", "ARG": "ar", "AUS": "au", "AUT": "at", "BEL": "be",
    "BIH": "ba", "BRA": "br", "CAN": "ca", "CIV": "ci", "COL": "co",
    "CPV": "cv", "CRC": None, "CRO": "hr", "CUW": "cw", "CZE": "cz",
    "DEN": "dk", "ECU": "ec", "EGY": "eg", "ENG": "gb-eng", "ESP": "es",
    "FRA": "fr", "GER": "de", "GHA": "gh", "HAI": "ht", "IRN": "ir",
    "IRQ": "iq", "ITA": "it", "JOR": "jo", "JPN": "jp", "KOR": "kr",
    "MAR": "ma", "MEX": "mx", "NED": "nl", "NOR": "no", "NZL": "nz",
    "PAN": "pa", "PAR": "py", "POR": "pt", "QAT": "qa", "RSA": "za",
    "SCO": "gb-sct", "SEN": "sn", "SRB": "rs", "SUI": "ch", "SWE": "se",
    "TUN": "tn", "TUR": "tr", "URU": "uy", "USA": "us", "UZB": "uz",
    "WAL": "gb-wls", "COD": "cd", "KSA": "sa",
}


def fetch_matches(api_key: str) -> list:
    """Fetch all WC 2026 matches from football-data.org."""
    url = f"{API_BASE}/competitions/{COMP_CODE}/matches?season={SEASON}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["matches"]


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def load_deterministic_predictions() -> dict:
    """Parse deterministic-data.ts for our pre-WC predictions, keyed by frozenset team pair."""
    if not DET_PATH.exists():
        return {}
    text = DET_PATH.read_text()
    m = re.search(r"DETERMINISTIC_DATA\s*=\s*(\{.*?\});\s*$", text, re.DOTALL)
    if not m:
        return {}
    data = json.loads(m.group(1))
    out = {}
    # Group matches
    for g in data.get("groups", []):
        for match in g.get("matches", []):
            h = match["home_team"]
            a = match["away_team"]
            out[(h, a)] = {
                "prob_home": match["prob_home"],
                "prob_draw": match["prob_draw"],
                "prob_away": match["prob_away"],
                "pred_home_goals": match["home_goals"],
                "pred_away_goals": match["away_goals"],
                "source": "pre_wc_deterministic",
            }
    # KO matches
    for round_name, matches in data.get("knockout", {}).items():
        for match in matches:
            t1 = match["team1"]
            t2 = match["team2"]
            out[(t1, t2)] = {
                "prob_home": match["prob_team1"],
                "prob_draw": match["prob_draw"],
                "prob_away": match["prob_team2"],
                "pred_home_goals": match["team1_goals"],
                "pred_away_goals": match["team2_goals"],
                "source": f"pre_wc_deterministic_{round_name}",
            }
    return out


def predicted_outcome(p: dict) -> str:
    """Return 'home'/'draw'/'away' from probabilities."""
    return max(["home", "draw", "away"], key=lambda k: p[f"prob_{k}"])


def stage_label(stage, group):
    """Map football-data stage to our shape."""
    if stage == "GROUP_STAGE":
        return {"stage": "group", "group": (group or "").replace("GROUP_", "") or None}
    s_map = {"LAST_32": "r32", "LAST_16": "r16", "QUARTER_FINALS": "qf",
             "SEMI_FINALS": "sf", "THIRD_PLACE": "tp", "FINAL": "final"}
    return {"stage": s_map.get(stage, stage.lower()), "group": None}


def build_match_record(fd_match: dict, predictions: dict) -> dict:
    home_name = normalize_team(fd_match["homeTeam"]["name"])
    away_name = normalize_team(fd_match["awayTeam"]["name"])
    home_tla = fd_match["homeTeam"].get("tla", "")
    away_tla = fd_match["awayTeam"].get("tla", "")

    stage_info = stage_label(fd_match["stage"], fd_match.get("group"))

    record = {
        "id": fd_match["id"],
        "kickoff_utc": fd_match["utcDate"],
        "status": fd_match["status"],  # SCHEDULED / TIMED / IN_PLAY / PAUSED / FINISHED / POSTPONED
        "matchday": fd_match.get("matchday"),
        "stage": stage_info["stage"],
        "group": stage_info["group"],
        "home": {
            "name": home_name,
            "tla": home_tla,
            "flag": TLA_TO_FLAG.get(home_tla),
        },
        "away": {
            "name": away_name,
            "tla": away_tla,
            "flag": TLA_TO_FLAG.get(away_tla),
        },
        "prediction": None,
        "actual": None,
    }

    # Attach our prediction if known
    pred = predictions.get((home_name, away_name))
    if pred:
        record["prediction"] = {
            "prob_home": pred["prob_home"],
            "prob_draw": pred["prob_draw"],
            "prob_away": pred["prob_away"],
            "score": {"home": pred["pred_home_goals"], "away": pred["pred_away_goals"]},
            "predicted_outcome": predicted_outcome(pred),
            "source": pred["source"],
        }

    # Attach actual result if finished
    score = fd_match.get("score", {})
    ft = (score or {}).get("fullTime") or {}
    if fd_match["status"] == "FINISHED" and ft.get("home") is not None:
        winner = score.get("winner")
        if winner == "HOME_TEAM":
            actual_outcome = "home"
        elif winner == "AWAY_TEAM":
            actual_outcome = "away"
        else:
            actual_outcome = "draw"

        record["actual"] = {
            "score": {"home": ft["home"], "away": ft["away"]},
            "outcome": actual_outcome,
        }

        # Outcome correctness + within-1-goal + upset detection
        if record["prediction"]:
            p = record["prediction"]
            record["actual"]["outcome_correct"] = (p["predicted_outcome"] == actual_outcome)
            record["actual"]["exact_score"] = (
                p["score"]["home"] == ft["home"] and p["score"]["away"] == ft["away"]
            )
            record["actual"]["within_1_goal"] = (
                abs(p["score"]["home"] - ft["home"]) <= 1
                and abs(p["score"]["away"] - ft["away"]) <= 1
            )
            # Upset: favorite (>=60%) was predicted, but lost
            fav_prob = max(p["prob_home"], p["prob_away"])
            if fav_prob >= 0.60:
                fav_outcome = "home" if p["prob_home"] >= p["prob_away"] else "away"
                record["actual"]["is_upset"] = (
                    fav_outcome != actual_outcome and actual_outcome != "draw"
                )
            else:
                record["actual"]["is_upset"] = False

    return record


def compute_stats(matches: list) -> dict:
    finished = [m for m in matches if m["actual"] and m["prediction"]]
    n = len(finished)
    if n == 0:
        return {"n_played": 0, "n_with_predictions": 0}

    n_correct = sum(1 for m in finished if m["actual"]["outcome_correct"])
    n_exact = sum(1 for m in finished if m["actual"]["exact_score"])
    n_within1 = sum(1 for m in finished if m["actual"]["within_1_goal"])
    n_upsets = sum(1 for m in finished if m["actual"].get("is_upset"))

    # Find biggest upset (favorite with highest prob who lost)
    biggest_upset = None
    for m in finished:
        if m["actual"].get("is_upset"):
            p = m["prediction"]
            fav_prob = max(p["prob_home"], p["prob_away"])
            fav_team = m["home"]["name"] if p["prob_home"] >= p["prob_away"] else m["away"]["name"]
            if biggest_upset is None or fav_prob > biggest_upset["fav_prob"]:
                biggest_upset = {
                    "match_id": m["id"],
                    "fav_team": fav_team,
                    "fav_prob": fav_prob,
                    "winner": m["home"]["name"] if m["actual"]["outcome"] == "home" else m["away"]["name"],
                }

    return {
        "n_played": n,
        "n_with_predictions": n,
        "n_correct_outcome": n_correct,
        "outcome_accuracy": round(n_correct / n, 4),
        "n_exact_score": n_exact,
        "exact_score_pct": round(n_exact / n, 4),
        "n_within_1_goal": n_within1,
        "within_1_goal_pct": round(n_within1 / n, 4),
        "n_upsets": n_upsets,
        "biggest_upset": biggest_upset,
    }


def main():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("ERROR: FOOTBALL_DATA_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching matches from football-data.org...")
    raw_matches = fetch_matches(api_key)
    print(f"  Got {len(raw_matches)} matches")

    print("Loading our pre-WC deterministic predictions...")
    predictions = load_deterministic_predictions()
    print(f"  {len(predictions)} predicted pairs loaded")

    matches = [build_match_record(m, predictions) for m in raw_matches]
    matches.sort(key=lambda x: x["kickoff_utc"])

    # Determine tournament phase
    n_finished = sum(1 for m in matches if m["status"] == "FINISHED")
    phase = "group_stage" if any(m["stage"] == "group" and m["status"] != "FINISHED" for m in matches) else "knockout"

    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tournament_phase": phase,
        "stats": compute_stats(matches),
        "matches": matches,
    }

    LIVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    LIVE_OUT.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    print(f"\nWrote {LIVE_OUT}")
    print(f"  {n_finished} finished, {len(matches) - n_finished} upcoming")
    if output["stats"].get("n_played"):
        s = output["stats"]
        print(f"  Outcome accuracy: {s['n_correct_outcome']}/{s['n_played']} = {s['outcome_accuracy']*100:.1f}%")
        print(f"  Exact score: {s['n_exact_score']}/{s['n_played']}")
        print(f"  Within 1 goal: {s['n_within_1_goal']}/{s['n_played']}")
        print(f"  Upsets: {s['n_upsets']}")


if __name__ == "__main__":
    main()
