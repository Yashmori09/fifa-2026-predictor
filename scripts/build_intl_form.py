"""
B5(v2)c — Recent international form for ALL 48 WC 2026 teams.

Computed from data/processed/matches_clean.csv (FIFA-recognized matches).
This is the universal-coverage fallback to StatsBomb features (which only
cover 39/48 teams via WC/Euro/Copa/AFCON tournaments).

Per team, looking at matches in the trailing window before WC 2026 opening day:
  intl_n_matches_2y         # matches in last 2 years (form recency)
  intl_goals_for_per_match  weighted by recency
  intl_goals_against_per_match
  intl_win_rate             share of wins
  intl_draw_rate
  intl_form_last10          avg points (W=3, D=1, L=0) over last 10 matches
  intl_competitive_pct      % matches that were qualifying/major (not friendlies)

Output: data/processed/team_intl_form_features.csv
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MC = ROOT / "data/processed/matches_clean.csv"
SQUADS = ROOT / "frontend/src/data/squad_players.json"
OUT = ROOT / "data/processed/team_intl_form_features.csv"

REF_DATE = pd.Timestamp(2026, 6, 11)  # WC opening day
WINDOW_DAYS = 730  # 2 years


def per_team_form(team: str, df: pd.DataFrame) -> dict:
    """All-matches involving this team in the window. df already filtered to window."""
    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()

    # Stack into one team-view dataframe
    home_view = pd.DataFrame({
        "date": home["date"],
        "goals_for": home["home_score"],
        "goals_against": home["away_score"],
        "is_competitive": home["tournament_category"] != "friendly",
    })
    away_view = pd.DataFrame({
        "date": away["date"],
        "goals_for": away["away_score"],
        "goals_against": away["home_score"],
        "is_competitive": away["tournament_category"] != "friendly",
    })
    games = pd.concat([home_view, away_view]).sort_values("date").reset_index(drop=True)

    if len(games) == 0:
        return {"intl_n_matches_2y": 0}

    games["outcome"] = (games["goals_for"] - games["goals_against"]).apply(
        lambda d: "W" if d > 0 else "L" if d < 0 else "D"
    )
    games["points"] = games["outcome"].map({"W": 3, "D": 1, "L": 0})

    # Recency weight: linear, more recent = higher
    days_ago = (REF_DATE - games["date"]).dt.days
    games["weight"] = (1 - days_ago / WINDOW_DAYS).clip(lower=0.1, upper=1.0)

    w = games["weight"]
    n = len(games)

    return {
        "intl_n_matches_2y": n,
        "intl_goals_for_per_match": round((games["goals_for"] * w).sum() / w.sum(), 3),
        "intl_goals_against_per_match": round((games["goals_against"] * w).sum() / w.sum(), 3),
        "intl_win_rate": round((games["outcome"] == "W").mean(), 3),
        "intl_draw_rate": round((games["outcome"] == "D").mean(), 3),
        "intl_form_last10": round(games.tail(10)["points"].mean(), 2) if n >= 5 else None,
        "intl_competitive_pct": round(games["is_competitive"].mean() * 100, 1),
    }


def main():
    mc = pd.read_csv(MC, parse_dates=["date"])
    print(f"Loaded {len(mc):,} matches from {mc['date'].min().date()} to {mc['date'].max().date()}")

    cutoff = REF_DATE - pd.Timedelta(days=WINDOW_DAYS)
    recent = mc[(mc["date"] >= cutoff) & (mc["date"] < REF_DATE)]
    print(f"In window ({cutoff.date()} → {REF_DATE.date()}): {len(recent):,} matches")

    import json as _json
    with open(SQUADS) as f:
        wc_teams = list(_json.load(f).keys())

    rows = []
    for team in wc_teams:
        rows.append({"team": team, **per_team_form(team, recent)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(out)} teams")
    print(f"  All teams have data: {(out['intl_n_matches_2y'] > 0).all()}\n")

    # Show ranked
    print("Top 5 by recent form (last-10 points):")
    print(out.nlargest(5, "intl_form_last10")[
        ["team", "intl_n_matches_2y", "intl_win_rate", "intl_form_last10",
         "intl_goals_for_per_match", "intl_goals_against_per_match"]
    ].to_string(index=False))

    print("\nBottom 5 by recent form:")
    print(out.nsmallest(5, "intl_form_last10")[
        ["team", "intl_n_matches_2y", "intl_win_rate", "intl_form_last10",
         "intl_goals_for_per_match", "intl_goals_against_per_match"]
    ].to_string(index=False))

    print("\nThe 9 SB-missing teams now have features:")
    missing = ["Bosnia and Herzegovina", "Cape Verde", "Curaçao", "Haiti", "Iraq",
               "Jordan", "New Zealand", "Norway", "Uzbekistan"]
    print(out[out["team"].isin(missing)][
        ["team", "intl_n_matches_2y", "intl_win_rate", "intl_form_last10",
         "intl_goals_for_per_match", "intl_goals_against_per_match"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
