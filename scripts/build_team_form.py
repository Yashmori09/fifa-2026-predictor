"""
B2 — Aggregate per-player trailing form features → per-team features.

Input:  data/processed/player_form_features.csv
Output: data/processed/team_form_features.csv

For each WC team, compute:
  Team totals & per-90 rates (minutes-weighted)
    - team_goals_per_90, team_assists_per_90, team_xg_per_90, team_xa_per_90
  Position-split rates
    - fwd_xg_per_90, mid_xg_per_90, def_xg_per_90 (attacking outputs by role)
    - fwd_goals_per_90, mid_goals_per_90, def_goals_per_90
  Top XI proxy (likely starters)
    - top11_minutes_avg, top11_goals_per_90, top11_xg_per_90
  Coverage metadata
    - n_players, n_with_xg, n_with_tm
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/player_form_features.csv"
OUT = ROOT / "data/processed/team_form_features.csv"


def per_90(numerator_col: str, df: pd.DataFrame, minutes_col: str = "tm_minutes") -> float:
    """Minutes-weighted per-90 rate."""
    mins = df[minutes_col].fillna(0).sum()
    if mins == 0:
        return None
    return round(float(df[numerator_col].fillna(0).sum()) * 90 / mins, 3)


def aggregate_team(team_df: pd.DataFrame) -> dict:
    out = {"team": team_df["team"].iloc[0]}

    # Coverage metadata
    out["n_players"] = len(team_df)
    out["n_with_tm"] = team_df["tm_id"].notna().sum()
    out["n_with_xg"] = team_df["us_xg"].notna().sum()

    # Team-level per-90 rates (using TM since it has 93% coverage)
    out["team_goals_per_90"] = per_90("tm_goals", team_df)
    out["team_assists_per_90"] = per_90("tm_assists", team_df)
    out["team_total_minutes"] = int(team_df["tm_minutes"].fillna(0).sum())
    out["team_avg_minutes_per_player"] = round(out["team_total_minutes"] / max(out["n_with_tm"], 1), 1)

    # xG rates (Big 5 only) — weighted by US minutes
    xg_df = team_df[team_df["us_xg"].notna()]
    if len(xg_df) > 0 and out["team_total_minutes"] > 0:
        out["team_xg_per_90"] = per_90("us_xg", xg_df, "us_minutes")
        out["team_np_xg_per_90"] = per_90("us_np_xg", xg_df, "us_minutes")
        out["team_xa_per_90"] = per_90("us_xa", xg_df, "us_minutes")
        # Coverage = (Understat minutes for the squad) / (TM minutes for the squad)
        us_min = float(xg_df["us_minutes"].sum())
        coverage = min(us_min / out["team_total_minutes"] * 100, 100.0)
        out["xg_coverage_minutes_pct"] = round(coverage, 1)
    else:
        out["team_xg_per_90"] = None
        out["team_np_xg_per_90"] = None
        out["team_xa_per_90"] = None
        out["xg_coverage_minutes_pct"] = 0.0

    # Position splits (using wc_pos: GK / DF / MF / FW)
    for pos, prefix in [("FW", "fwd"), ("MF", "mid"), ("DF", "def"), ("GK", "gk")]:
        sub = team_df[team_df["wc_pos"] == pos]
        out[f"{prefix}_n"] = len(sub)
        if len(sub) > 0:
            out[f"{prefix}_goals_per_90"] = per_90("tm_goals", sub)
            out[f"{prefix}_assists_per_90"] = per_90("tm_assists", sub)
            xg_sub = sub[sub["us_xg"].notna()]
            out[f"{prefix}_xg_per_90"] = per_90("us_xg", xg_sub, "us_minutes") if len(xg_sub) else None
        else:
            out[f"{prefix}_goals_per_90"] = None
            out[f"{prefix}_assists_per_90"] = None
            out[f"{prefix}_xg_per_90"] = None

    # Top-11 proxy: top 11 players by trailing minutes (likely starters)
    top11 = team_df.nlargest(11, "tm_minutes")
    out["top11_total_minutes"] = int(top11["tm_minutes"].fillna(0).sum())
    out["top11_avg_minutes"] = round(out["top11_total_minutes"] / 11, 1)
    out["top11_goals_per_90"] = per_90("tm_goals", top11)
    out["top11_assists_per_90"] = per_90("tm_assists", top11)
    top11_xg = top11[top11["us_xg"].notna()]
    out["top11_xg_per_90"] = per_90("us_xg", top11_xg, "us_minutes") if len(top11_xg) else None

    return out


def main():
    print(f"Loading {SRC}...")
    df = pd.read_csv(SRC)
    print(f"  {len(df)} players, {df['team'].nunique()} teams")

    rows = []
    for team in sorted(df["team"].unique()):
        team_df = df[df["team"] == team]
        rows.append(aggregate_team(team_df))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(out)} teams × {len(out.columns)} features\n")

    # Show ranked by top11 xG/90 (the most predictive single metric)
    print("Top 10 teams by top11_xg_per_90 (xG from likely starting XI):")
    ranked = out[out["top11_xg_per_90"].notna()].nlargest(10, "top11_xg_per_90")
    print(ranked[["team", "top11_avg_minutes", "team_xg_per_90", "top11_xg_per_90", "xg_coverage_minutes_pct"]].to_string(index=False))
    print("\nBottom 5 by xG coverage (rely on team-level fallback):")
    print(out.nsmallest(5, "xg_coverage_minutes_pct")[["team", "n_players", "n_with_xg", "xg_coverage_minutes_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
