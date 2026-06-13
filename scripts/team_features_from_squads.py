"""
Compute team-level squad features for 2026 from frontend/src/data/squad_players.json.

This is a self-contained version of the original build_team_features.py logic that
runs without the EA FC raw CSVs. Output mirrors the team feature schema used by the
trained Phase 2 model.

Output: data/processed/team_features_2026.csv
"""

import csv
import json
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent
SQUADS_PATH = ROOT / "frontend/src/data/squad_players.json"
OUT_PATH = ROOT / "data/processed/team_features_2026.csv"

FEATURE_COLS = [
    "team", "year", "n_players", "n_rated", "coverage",
    # Core quality
    "squad_avg_overall", "squad_median_overall", "squad_std_overall",
    "squad_top3_avg", "squad_bottom5_avg",
    # Positional
    "gk_avg", "def_avg", "mid_avg", "fwd_avg",
    "strongest_unit", "weakest_unit",
    # Market value
    "squad_total_value", "squad_avg_value",
    # Squad profile
    "squad_avg_age", "squad_avg_potential_gap", "squad_avg_caps",
    # Big 6 face stats (outfield only)
    "team_pace", "team_shooting", "team_passing",
    "team_dribbling", "team_defending", "team_physic",
]


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def safe_median(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def safe_stdev(xs):
    xs = [x for x in xs if x is not None]
    return stdev(xs) if len(xs) >= 2 else 0


def topn_mean(xs, n):
    xs = sorted([x for x in xs if x is not None], reverse=True)
    return mean(xs[:n]) if xs else None


def bottomn_mean(xs, n):
    xs = sorted([x for x in xs if x is not None])
    return mean(xs[:n]) if xs else None


def compute_team(team, players):
    """Compute 22 features for a single team's squad."""
    out = {"team": team, "year": 2026, "n_players": len(players)}

    rated = [p for p in players if p.get("ovr") is not None]
    out["n_rated"] = len(rated)
    out["coverage"] = round(len(rated) / len(players), 3) if players else 0

    if not rated:
        return out  # all None for the rest

    ovrs = [p["ovr"] for p in rated]
    pots = [p.get("pot") for p in rated]
    ages = [p.get("age") for p in rated]
    caps = [p.get("caps") for p in rated]
    vals = [p.get("val") for p in rated]

    out["squad_avg_overall"] = round(safe_mean(ovrs), 2)
    out["squad_median_overall"] = round(safe_median(ovrs), 2)
    out["squad_std_overall"] = round(safe_stdev(ovrs), 2)
    out["squad_top3_avg"] = round(topn_mean(ovrs, 3), 2)
    out["squad_bottom5_avg"] = round(bottomn_mean(ovrs, 5), 2)

    # Positional averages — use the player's position (GK/DF/MF/FW)
    by_pos = {"GK": [], "DF": [], "MF": [], "FW": []}
    for p in rated:
        pos = p.get("pos", "")
        if pos in by_pos:
            by_pos[pos].append(p["ovr"])

    out["gk_avg"] = round(safe_mean(by_pos["GK"]), 2) if by_pos["GK"] else None
    out["def_avg"] = round(safe_mean(by_pos["DF"]), 2) if by_pos["DF"] else None
    out["mid_avg"] = round(safe_mean(by_pos["MF"]), 2) if by_pos["MF"] else None
    out["fwd_avg"] = round(safe_mean(by_pos["FW"]), 2) if by_pos["FW"] else None

    unit_avgs = {k: v for k, v in [("gk", out["gk_avg"]), ("def", out["def_avg"]),
                                    ("mid", out["mid_avg"]), ("fwd", out["fwd_avg"])] if v}
    if unit_avgs:
        out["strongest_unit"] = round(max(unit_avgs.values()), 2)
        out["weakest_unit"] = round(min(unit_avgs.values()), 2)

    # Market value
    rated_vals = [v for v in vals if v is not None]
    out["squad_total_value"] = sum(rated_vals) if rated_vals else None
    out["squad_avg_value"] = round(safe_mean(rated_vals), 0) if rated_vals else None

    # Squad profile
    out["squad_avg_age"] = round(safe_mean(ages), 2) if any(a is not None for a in ages) else None
    out["squad_avg_caps"] = round(safe_mean(caps), 2) if any(c is not None for c in caps) else None
    pot_gaps = [pt - ov for pt, ov in zip(pots, ovrs) if pt is not None and ov is not None]
    out["squad_avg_potential_gap"] = round(safe_mean(pot_gaps), 2) if pot_gaps else None

    # Big 6 face stats (outfield only — GKs don't have these)
    outfield = [p for p in rated if p.get("pos") != "GK"]
    for col, key in [("team_pace", "pac"), ("team_shooting", "sho"),
                     ("team_passing", "pas"), ("team_dribbling", "dri"),
                     ("team_defending", "defe"), ("team_physic", "phy")]:
        vs = [p.get(key) for p in outfield]
        m = safe_mean(vs)
        out[col] = round(m, 2) if m is not None else None

    return out


def main():
    with open(SQUADS_PATH) as f:
        squads = json.load(f)

    rows = []
    for team in sorted(squads.keys()):
        row = compute_team(team, squads[team])
        rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FEATURE_COLS})

    print(f"Wrote {OUT_PATH} ({len(rows)} teams)")
    print(f"\n{'Team':<22} {'N':>3} {'Cov%':>5} {'Avg':>5} {'Top3':>5} {'GK':>5} {'DEF':>5} {'MID':>5} {'FWD':>5}")
    print("-" * 75)
    for r in rows:
        avg = r.get("squad_avg_overall", "—")
        top = r.get("squad_top3_avg", "—")
        gk = r.get("gk_avg") or "—"
        df = r.get("def_avg") or "—"
        md = r.get("mid_avg") or "—"
        fw = r.get("fwd_avg") or "—"
        print(f"{r['team']:<22} {r['n_rated']:>3} {int(r['coverage']*100):>4}% {avg!s:>5} {top!s:>5} {gk!s:>5} {df!s:>5} {md!s:>5} {fw!s:>5}")


if __name__ == "__main__":
    main()
