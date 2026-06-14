"""
B3 — Team chemistry features.

For each WC team:
  same_club_top1_pct      — share of squad at the single top club
  same_club_top3_pct      — share of squad at top 3 clubs combined
  n_unique_clubs          — squad spread across N clubs
  avg_intl_caps           — average international caps across squad (experience together proxy)
  avg_squad_age           — average squad age

NOTE: We initially tried "shared international starts per pair" computed
from TM appearances, but TM's appearances dataset is club-focused — it
covers only AFCN among international competitions. So that feature
collapsed to zeros for most teams. avg_intl_caps is the available proxy.

Output: data/processed/team_chemistry_features.csv
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQUADS = ROOT / "frontend/src/data/squad_players.json"
RES = ROOT / "data/processed/wc_tm_resolution.csv"
TMP = ROOT / "data/raw/transfermarkt/players.csv.gz"
OUT = ROOT / "data/processed/team_chemistry_features.csv"
REF_DATE = datetime(2026, 6, 11)  # WC opening day


def main():
    with open(SQUADS) as f:
        squads = json.load(f)
    resolver = pd.read_csv(RES)
    tm = pd.read_csv(TMP, compression="gzip", low_memory=False, parse_dates=["date_of_birth"])

    # Build name → DOB lookup via TM
    name_to_dob = {}
    name_to_tm = dict(zip(resolver["name"], resolver["tm_id"]))
    for name, tm_id in name_to_tm.items():
        if pd.notna(tm_id):
            row = tm[tm["player_id"] == int(tm_id)]
            if len(row) > 0 and pd.notna(row.iloc[0]["date_of_birth"]):
                name_to_dob[name] = row.iloc[0]["date_of_birth"]

    rows = []
    for team, players in squads.items():
        n = len(players)
        clubs = [p.get("club", "") for p in players if p.get("club")]
        club_counts = Counter(clubs)
        top1 = max(club_counts.values()) if club_counts else 0
        top3 = sum(sorted(club_counts.values(), reverse=True)[:3])
        n_unique = len(club_counts)

        caps = [p.get("caps", 0) for p in players if isinstance(p.get("caps"), (int, float))]
        ages = []
        for p in players:
            dob = name_to_dob.get(p["name"])
            if dob is not None:
                age = (REF_DATE - dob.to_pydatetime()).days / 365.25
                ages.append(age)

        rows.append({
            "team": team,
            "n_squad": n,
            "same_club_top1_pct": round(top1 / n * 100, 1),
            "same_club_top3_pct": round(top3 / n * 100, 1),
            "n_unique_clubs": n_unique,
            "avg_intl_caps": round(sum(caps) / len(caps), 1) if caps else 0,
            "avg_squad_age": round(sum(ages) / len(ages), 1) if ages else 0,
            "n_with_dob": len(ages),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} teams × {len(out.columns)} features\n")

    print("Top 5 same-club concentration (highest = most club-mate chemistry):")
    print(out.nlargest(5, "same_club_top3_pct")[["team","same_club_top1_pct","same_club_top3_pct"]].to_string(index=False))
    print("\nTop 5 avg international caps (most veteran squads):")
    print(out.nlargest(5, "avg_intl_caps")[["team","avg_intl_caps","avg_squad_age"]].to_string(index=False))
    print("\nBottom 5 avg caps (youngest / least experienced together):")
    print(out.nsmallest(5, "avg_intl_caps")[["team","avg_intl_caps","avg_squad_age"]].to_string(index=False))


if __name__ == "__main__":
    main()
