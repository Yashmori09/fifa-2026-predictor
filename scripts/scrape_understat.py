"""
Pull per-player per-season stats from Understat via soccerdata.

6 leagues × 3 seasons = 18 league-season combos.
Soccer data caches everything in ~/soccerdata/data/Understat/

Output: data/processed/understat_players.csv
"""
import sys
import warnings
import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

warnings.filterwarnings("ignore")
logging.getLogger("soccerdata").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/understat_players.csv"

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "NED-Eredivisie",
]
SEASONS = ["2324", "2425", "2526"]


def main():
    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            print(f"  fetching {league}  season {season}...", end=" ", flush=True)
            try:
                u = sd.Understat(leagues=league, seasons=season)
                df = u.read_player_season_stats()
                df = df.reset_index()  # flatten multi-index
                frames.append(df)
                print(f"{len(df)} players")
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {str(e)[:100]}")

    if not frames:
        print("Nothing fetched. Aborting.")
        sys.exit(1)

    all_players = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_players.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(f"  Total rows: {len(all_players)}")
    print(f"  Unique players (any season): {all_players['player'].nunique()}")
    print(f"  Leagues: {sorted(all_players['league'].unique())}")
    print(f"  Seasons: {sorted(all_players['season'].unique())}")


if __name__ == "__main__":
    main()
