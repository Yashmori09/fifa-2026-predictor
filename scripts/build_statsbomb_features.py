"""
B5(v2) — Derive per-team features from StatsBomb Open Data event-level data
for international tournaments (WC 2018/2022, Euro 2020/2024, Copa Am 2024,
AFCON 2023).

For each WC 2026 team, compute (across all matches found):
  sb_n_matches               coverage count
  sb_xg_for_per_match        avg xG generated
  sb_xg_against_per_match    avg xG conceded ← key new signal
  sb_xg_set_piece_share      % of own xG from FKs + Corners
  sb_def_overperformance     xG conceded − goals conceded, per match
                             positive = D+GK saved more than chance quality predicted
  sb_att_overperformance     goals scored − xG, per match
                             positive = clinical finishing beyond chance quality
  sb_goals_for_per_match
  sb_goals_against_per_match
  sb_avg_year                weighted-average tournament year (recency proxy)

Output: data/processed/team_statsbomb_features.csv (48 WC teams; NaN where
no SB data exists for that team)
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MDIR = ROOT / "data/raw/statsbomb/matches"
EDIR = ROOT / "data/raw/statsbomb/events"
SQUADS = ROOT / "frontend/src/data/squad_players.json"
OUT = ROOT / "data/processed/team_statsbomb_features.csv"

# StatsBomb name → our WC2026 name
NAME_FIX = {
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Republic of Ireland": "Ireland",
}

# Tournament metadata for recency weighting
TOURNAMENT_INFO = {
    "43_106": ("WC 2022", 2022),
    "43_3": ("WC 2018", 2018),
    "55_282": ("Euro 2024", 2024),
    "55_43": ("Euro 2020", 2021),  # played 2021 due to covid
    "223_282": ("Copa America 2024", 2024),
    "1267_107": ("AFCON 2023", 2024),  # played early 2024
}

# Shot on-target outcomes (used for PSxG approximation)
ON_TARGET = {"Goal", "Saved", "Saved To Post"}
# Set-piece shot types from StatsBomb 'shot.type.name'
SET_PIECE_TYPES = {"Free Kick", "Corner"}


def normalize_team(name: str) -> str:
    return NAME_FIX.get(name, name)


def process_match(match: dict, events: list) -> dict:
    """Return per-team stats for this single match."""
    home = normalize_team(match["home_team"]["home_team_name"])
    away = normalize_team(match["away_team"]["away_team_name"])
    h_score = match["home_score"]
    a_score = match["away_score"]

    stats = {
        home: {"xg_for": 0, "xg_against": 0, "sp_xg_for": 0,
               "on_target_xg_against": 0, "goals_against": a_score, "goals_for": h_score},
        away: {"xg_for": 0, "xg_against": 0, "sp_xg_for": 0,
               "on_target_xg_against": 0, "goals_against": h_score, "goals_for": a_score},
    }

    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        shot = ev.get("shot", {})
        xg = shot.get("statsbomb_xg", 0) or 0
        shot_type = shot.get("type", {}).get("name", "")
        outcome = shot.get("outcome", {}).get("name", "")
        if shot_type == "Penalty":
            # exclude penalties from xG (they distort everything)
            continue

        team = normalize_team(ev["team"]["name"])
        if team not in stats:
            continue
        opponent = home if team == away else away

        stats[team]["xg_for"] += xg
        stats[opponent]["xg_against"] += xg
        if shot_type in SET_PIECE_TYPES:
            stats[team]["sp_xg_for"] += xg
        if outcome in ON_TARGET:
            stats[opponent]["on_target_xg_against"] += xg

    return stats


def main():
    # Build a flat list of (team, tournament_year, per-match-stats)
    rows = []

    for tfile in sorted(MDIR.glob("*.json")):
        tkey = tfile.stem
        tname, year = TOURNAMENT_INFO.get(tkey, (tkey, 2020))
        matches = json.loads(tfile.read_text())
        print(f"Processing {tname} ({len(matches)} matches)...")

        for m in matches:
            mid = m["match_id"]
            efile = EDIR / f"{mid}.json"
            if not efile.exists():
                continue
            events = json.loads(efile.read_text())
            per_team = process_match(m, events)
            for team, st in per_team.items():
                rows.append({
                    "team": team,
                    "tournament": tname,
                    "year": year,
                    **st,
                })

    df = pd.DataFrame(rows)
    print(f"\nTotal team-match rows: {len(df)}")
    print(f"Unique teams in SB data: {df['team'].nunique()}")

    # Aggregate per team. Recency weight: linear 0.5 (2018) → 1.0 (2024)
    today_year = 2026
    df["weight"] = ((df["year"] - 2018) / (today_year - 2018)).clip(lower=0.3, upper=1.0)

    def weighted_avg(g, col):
        w = g["weight"]
        return (g[col] * w).sum() / w.sum() if w.sum() > 0 else None

    agg = []
    for team, g in df.groupby("team"):
        n = len(g)
        xg_for = weighted_avg(g, "xg_for")
        xg_against = weighted_avg(g, "xg_against")
        goals_for = weighted_avg(g, "goals_for")
        goals_against = weighted_avg(g, "goals_against")
        agg.append({
            "team": team,
            "sb_n_matches": n,
            "sb_xg_for_per_match": round(xg_for, 3),
            "sb_xg_against_per_match": round(xg_against, 3),
            "sb_xg_set_piece_share": round(g["sp_xg_for"].sum() / max(g["xg_for"].sum(), 0.001), 3),
            "sb_def_overperformance": round(xg_against - goals_against, 3),
            "sb_att_overperformance": round(goals_for - xg_for, 3),
            "sb_goals_for_per_match": round(goals_for, 3),
            "sb_goals_against_per_match": round(goals_against, 3),
            "sb_avg_year": round((g["year"] * g["weight"]).sum() / g["weight"].sum(), 1),
            "sb_tournaments": ", ".join(sorted(g["tournament"].unique())),
        })

    sb_features = pd.DataFrame(agg)

    # Join against WC 2026 squad list — keep all 48 teams, NaN where no SB data
    with open(SQUADS) as f:
        wc_teams = list(json.load(f).keys())
    base = pd.DataFrame({"team": wc_teams})
    out = base.merge(sb_features, on="team", how="left")
    out["sb_n_matches"] = out["sb_n_matches"].fillna(0).astype(int)

    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(out)} teams")
    print(f"  with SB data: {(out['sb_n_matches'] > 0).sum()}/{len(out)}")
    print(f"  without (NaN features): {(out['sb_n_matches'] == 0).sum()}\n")

    # Top 5 attacking by xG/match
    print("Top 5 by SB xG for / match:")
    print(out.nlargest(5, "sb_xg_for_per_match")[
        ["team", "sb_n_matches", "sb_xg_for_per_match", "sb_xg_against_per_match", "sb_goals_for_per_match"]
    ].to_string(index=False))

    print("\nTop 5 by defensive strength (lowest xG against /match, min 5 matches):")
    qual = out[out["sb_n_matches"] >= 5]
    print(qual.nsmallest(5, "sb_xg_against_per_match")[
        ["team", "sb_n_matches", "sb_xg_against_per_match", "sb_goals_against_per_match"]
    ].to_string(index=False))

    print("\nTop 5 D+GK overperformers (conceded fewer goals than xG allowed, min 5 matches):")
    print(qual.nlargest(5, "sb_def_overperformance")[
        ["team", "sb_n_matches", "sb_xg_against_per_match", "sb_goals_against_per_match", "sb_def_overperformance"]
    ].to_string(index=False))

    print("\nTop 5 clinical attackers (scored more than xG, min 5 matches):")
    print(qual.nlargest(5, "sb_att_overperformance")[
        ["team", "sb_n_matches", "sb_xg_for_per_match", "sb_goals_for_per_match", "sb_att_overperformance"]
    ].to_string(index=False))

    print("\nTeams WITHOUT StatsBomb data (will have NaN features):")
    for t in out[out["sb_n_matches"] == 0]["team"].tolist():
        print(f"  - {t}")


if __name__ == "__main__":
    main()
