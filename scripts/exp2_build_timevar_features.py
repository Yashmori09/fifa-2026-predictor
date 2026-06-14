"""
Experiment 2 — Build per-match-date time-varying features from matches_clean.

For every match in train_dc_v2 + test_dc_v2, compute 16 features (8 per team)
using ONLY matches BEFORE that match's date — so no leakage into the future.

Features per team (suffixed _home or _away):
  elo_momentum_6mo       Δ team's effective ELO over prior 6 months
                         (approximated as avg ELO before recent matches)
  sos_adjusted_pts_10    points earned in last 10 weighted by opponent ELO
                         (avg opponent ELO * win rate; high = strong wins)
  goals_vs_strong_10     goals scored against top-quality opp (ELO ≥ 1700) in last 10
  conceded_vs_strong_10  goals conceded vs top-quality opp in last 10
  streak_unbeaten        current unbeaten run going into this match
  volatility_gd_10       std-dev of goal-difference over last 10 matches
  tourn_xp_4y            major-tournament matches played in last 4 years
  days_since_competitive days since last non-friendly match

We compute on the FULL matches_clean (not just our train pool) so the rolling
windows have proper history for early-training matches too.

Output: data/processed/timevar_features.csv keyed by (date, team, side)
        plus a merged variant for direct join: data/processed/timevar_match_features.csv

Then a wrapper join in exp2_eval.py uses these in the model.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"

MC = PROC / "matches_clean.csv"
OUT_TEAM = PROC / "timevar_team_features.csv"
OUT_MATCH = PROC / "timevar_match_features.csv"

STRONG_ELO = 1700  # threshold for "quality opponent"
MAJOR_TOURNAMENTS = {"continental_final", "world_cup"}


def make_team_match_view(mc: pd.DataFrame) -> pd.DataFrame:
    """Stack home + away into one team-perspective view.
    Each row = one team in one match."""
    home = mc[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "tournament_category", "home_elo_before", "away_elo_before"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against",
                    "tournament", "tournament_category", "team_elo", "opp_elo"]
    home["side"] = "home"

    away = mc[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "tournament_category", "home_elo_before", "away_elo_before"]].copy()
    away.columns = ["date", "opponent", "team", "goals_against", "goals_for",
                    "tournament", "tournament_category", "opp_elo", "team_elo"]
    away["side"] = "away"

    view = pd.concat([home, away], ignore_index=True)
    view = view.sort_values(["team", "date"]).reset_index(drop=True)
    view["goal_diff"] = view["goals_for"] - view["goals_against"]
    view["outcome"] = view["goal_diff"].apply(
        lambda d: 3 if d > 0 else 1 if d == 0 else 0)  # W=3, D=1, L=0 (points)
    view["unbeaten"] = (view["goal_diff"] >= 0).astype(int)
    return view


def compute_team_features(view: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute backwards-rolling features using only prior matches."""
    out_rows = []
    grouped = view.groupby("team", sort=False)
    for team, g in grouped:
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        for i in range(n):
            current_date = g.iloc[i]["date"]
            prior = g.iloc[:i]  # everything before current match
            if len(prior) == 0:
                rec = {"date": current_date, "team": team}
                # All features default to neutral / NaN
                for f in ["elo_momentum_6mo", "sos_adjusted_pts_10",
                          "goals_vs_strong_10", "conceded_vs_strong_10",
                          "streak_unbeaten", "volatility_gd_10",
                          "tourn_xp_4y", "days_since_competitive"]:
                    rec[f] = np.nan
                out_rows.append(rec)
                continue

            # 1. ELO momentum (last 6mo): mean ELO in last 6mo minus 6-12mo ago
            cutoff_6mo = current_date - pd.Timedelta(days=180)
            cutoff_12mo = current_date - pd.Timedelta(days=365)
            recent_elo = prior[prior["date"] >= cutoff_6mo]["team_elo"]
            older_elo = prior[(prior["date"] < cutoff_6mo) & (prior["date"] >= cutoff_12mo)]["team_elo"]
            elo_mom = (recent_elo.mean() - older_elo.mean()
                       if len(recent_elo) >= 2 and len(older_elo) >= 2
                       else np.nan)

            # 2-3-4. last-10 window features
            last_n = prior.tail(10)
            if len(last_n) >= 3:
                # SOS-adjusted points: weight each match's points by opponent ELO
                # higher value = points earned against stronger opponents
                weights = last_n["opp_elo"].values
                w_sum = weights.sum()
                sos_pts = float((last_n["outcome"] * weights).sum() / w_sum) if w_sum > 0 else np.nan

                # Goals vs strong opponents only
                strong = last_n[last_n["opp_elo"] >= STRONG_ELO]
                if len(strong) >= 1:
                    g_vs_strong = strong["goals_for"].sum() / len(strong)
                    c_vs_strong = strong["goals_against"].sum() / len(strong)
                else:
                    g_vs_strong = np.nan
                    c_vs_strong = np.nan

                # Volatility of goal difference
                vol_gd = float(last_n["goal_diff"].std())
            else:
                sos_pts = g_vs_strong = c_vs_strong = vol_gd = np.nan

            # 5. Unbeaten streak going into this match
            streak = 0
            for j in range(len(prior) - 1, -1, -1):
                if prior.iloc[j]["unbeaten"] == 1:
                    streak += 1
                else:
                    break

            # 6. Major tournament experience last 4 years
            cutoff_4yr = current_date - pd.Timedelta(days=365 * 4)
            tourn_xp = int(prior[(prior["date"] >= cutoff_4yr) &
                                  (prior["tournament_category"].isin(MAJOR_TOURNAMENTS))].shape[0])

            # 7. Days since last competitive match
            comp = prior[prior["tournament_category"] != "friendly"]
            days_since = ((current_date - comp["date"].max()).days
                          if len(comp) > 0 else np.nan)

            out_rows.append({
                "date": current_date, "team": team,
                "elo_momentum_6mo": elo_mom,
                "sos_adjusted_pts_10": sos_pts,
                "goals_vs_strong_10": g_vs_strong,
                "conceded_vs_strong_10": c_vs_strong,
                "streak_unbeaten": streak,
                "volatility_gd_10": vol_gd,
                "tourn_xp_4y": tourn_xp,
                "days_since_competitive": days_since,
            })
    return pd.DataFrame(out_rows)


def main():
    print("=" * 80)
    print("EXPERIMENT 2 — Build time-varying contextual features")
    print("=" * 80)

    print("\n[Load] matches_clean.csv...")
    mc = pd.read_csv(MC, parse_dates=["date"])
    mc = mc[mc["home_score"].notna() & mc["away_score"].notna()].copy()

    # We need home_elo_before / away_elo_before. Confirm they're in matches_clean.
    if "home_elo_before" not in mc.columns:
        # Compute from features_matrix instead
        print("  elo columns missing in matches_clean — joining from features_matrix...")
        fm = pd.read_csv(PROC / "features_matrix.csv", parse_dates=["date"])
        mc = mc.merge(
            fm[["date", "home_team", "away_team", "home_elo_before", "away_elo_before"]],
            on=["date", "home_team", "away_team"], how="left",
        )
    print(f"  loaded {len(mc):,} matches with ELO")

    print("\n[Build] team-match view...")
    view = make_team_match_view(mc)
    print(f"  {len(view):,} team-matches across {view['team'].nunique()} teams")

    print("\n[Compute] backwards-rolling features (this is the slow part)...")
    t0 = time.time()
    feats = compute_team_features(view)
    print(f"  {len(feats):,} feature rows in {time.time()-t0:.0f}s")

    # Sanity check
    print("\n[Sanity] Brazil's last 5 feature rows:")
    print(feats[feats["team"] == "Brazil"].tail(5).to_string(index=False))

    feats.to_csv(OUT_TEAM, index=False)
    print(f"\n  Saved per-team file: {OUT_TEAM.name}")

    # Now build a per-match version: home and away features side by side
    print("\n[Merge] building per-match feature file...")
    home_view = feats.rename(columns={"team": "home_team", **{
        c: f"home_{c}" for c in feats.columns if c not in ("date", "team")
    }})
    away_view = feats.rename(columns={"team": "away_team", **{
        c: f"away_{c}" for c in feats.columns if c not in ("date", "team")
    }})
    base = mc[["date", "home_team", "away_team"]].drop_duplicates()
    merged = base.merge(home_view, on=["date", "home_team"], how="left")
    merged = merged.merge(away_view, on=["date", "away_team"], how="left")
    merged.to_csv(OUT_MATCH, index=False)
    print(f"  Saved per-match file: {OUT_MATCH.name} ({len(merged):,} rows × {len(merged.columns)} cols)")

    # Show coverage
    feat_cols = [c for c in merged.columns if c.startswith("home_") or c.startswith("away_")]
    print(f"\n  Feature coverage in merged file:")
    for c in feat_cols[:8]:
        coverage = merged[c].notna().mean() * 100
        print(f"    {c:<35} {coverage:5.1f}% populated")


if __name__ == "__main__":
    main()
