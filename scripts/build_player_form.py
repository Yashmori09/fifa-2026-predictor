"""
Build trailing-12-month player form features for any (player, reference_date).

For each WC squad player:
  - From Transfermarkt appearances: goals, assists, minutes_played, matches
    aggregated over the 365 days before ref_date
  - From Understat (Big-5 only): xG, xA, np_xG using the most recent
    *completed* season at ref_date

A test mode runs on 3 known players (Vinicius, Taremi, Davies) and prints
their stats as of 2026-06-01 to verify the logic before scaling.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Load data ──────────────────────────────────────────────────


def load_appearances(filter_player_ids=None):
    """TM appearances since 2012 — 41MB compressed, filter early."""
    print("Loading appearances...", end=" ", flush=True)
    app = pd.read_csv(ROOT / "data/raw/transfermarkt/appearances.csv.gz",
                     compression="gzip",
                     parse_dates=["date"])
    if filter_player_ids is not None:
        app = app[app["player_id"].isin(filter_player_ids)]
    print(f"{len(app):,} rows")
    return app


def load_understat():
    """Per-season Big-5 stats (xG/xA)."""
    us = pd.read_csv(ROOT / "data/processed/understat_players.csv")
    return us


def load_resolver():
    return pd.read_csv(ROOT / "data/processed/wc_tm_resolution.csv")


def load_understat_resolver():
    return pd.read_csv(ROOT / "data/processed/wc_understat_resolution.csv")


# ── Form computation ───────────────────────────────────────────


def compute_tm_form(app: pd.DataFrame, player_id: int, ref_date: datetime, window_days: int = 365) -> dict:
    """Trailing-window stats from TM appearances."""
    start = ref_date - timedelta(days=window_days)
    window = app[(app["player_id"] == player_id) &
                 (app["date"] < ref_date) &
                 (app["date"] >= start)]
    matches = len(window)
    if matches == 0:
        return {"tm_matches": 0, "tm_minutes": 0, "tm_goals": 0, "tm_assists": 0,
                "tm_goals_per_90": None, "tm_assists_per_90": None,
                "tm_minutes_per_match": None}
    minutes = window["minutes_played"].sum()
    goals = window["goals"].sum()
    assists = window["assists"].sum()
    return {
        "tm_matches": int(matches),
        "tm_minutes": int(minutes),
        "tm_goals": int(goals),
        "tm_assists": int(assists),
        "tm_goals_per_90": round(goals * 90 / minutes, 3) if minutes else None,
        "tm_assists_per_90": round(assists * 90 / minutes, 3) if minutes else None,
        "tm_minutes_per_match": round(minutes / matches, 1) if matches else None,
    }


def compute_understat_form(us: pd.DataFrame, understat_name: str, ref_date: datetime) -> dict:
    """
    Trailing season xG/xA. Understat is per-season, so we use the most
    recent *completed* season at ref_date.

    Season '2425' covers Aug 2024 → May 2025. So at ref_date June 2026,
    we use 2526 season (just completed). At ref_date Jan 2025 mid-season,
    we use 2324 (last completed).
    """
    year = ref_date.year
    month = ref_date.month
    if month >= 6:
        ssn = f"{(year-1) % 100:02d}{year % 100:02d}"
    else:
        ssn = f"{(year-2) % 100:02d}{(year-1) % 100:02d}"
    rows = us[(us["player"] == understat_name) &
              (us["season"] == int(ssn))]
    if len(rows) == 0:
        return {"us_season": None, "us_matches": 0, "us_minutes": 0,
                "us_xg": None, "us_xa": None, "us_np_xg": None,
                "us_xg_per_90": None, "us_xa_per_90": None, "us_np_xg_per_90": None}
    # If multiple (player switched club mid-season), sum
    r = rows.agg({"matches": "sum", "minutes": "sum", "goals": "sum",
                   "xg": "sum", "xa": "sum", "np_xg": "sum"})
    minutes = float(r["minutes"])
    return {
        "us_season": ssn,
        "us_matches": int(r["matches"]),
        "us_minutes": int(minutes),
        "us_xg": round(float(r["xg"]), 2),
        "us_xa": round(float(r["xa"]), 2),
        "us_np_xg": round(float(r["np_xg"]), 2),
        "us_xg_per_90": round(float(r["xg"]) * 90 / minutes, 3) if minutes else None,
        "us_xa_per_90": round(float(r["xa"]) * 90 / minutes, 3) if minutes else None,
        "us_np_xg_per_90": round(float(r["np_xg"]) * 90 / minutes, 3) if minutes else None,
    }


def compute_player_form(app, us, resolver, us_resolver, wc_name: str, ref_date: datetime) -> dict:
    """Combined form for a WC player at a date."""
    row = resolver[resolver["name"] == wc_name]
    if len(row) == 0 or not row.iloc[0]["matched"]:
        return {"player": wc_name, "matched": False}

    tm_id = int(row.iloc[0]["tm_id"])
    out = {"player": wc_name, "tm_id": tm_id, "ref_date": ref_date.date(), "matched": True}
    out.update(compute_tm_form(app, tm_id, ref_date))

    # Look up Understat name (if mapped) and compute its form
    us_row = us_resolver[us_resolver["name"] == wc_name]
    if len(us_row) > 0 and pd.notna(us_row.iloc[0]["matched"]):
        out.update(compute_understat_form(us, us_row.iloc[0]["matched"], ref_date))
    else:
        out.update({"us_season": None, "us_matches": 0, "us_minutes": 0,
                    "us_xg": None, "us_xa": None, "us_np_xg": None,
                    "us_xg_per_90": None, "us_xa_per_90": None, "us_np_xg_per_90": None})
    return out


# ── Test ──────────────────────────────────────────────────────


def main():
    import json as _json
    test_mode = "--test" in sys.argv

    resolver = load_resolver()
    us_resolver = load_understat_resolver()
    wc_ids = set(resolver[resolver["matched"]]["tm_id"].dropna().astype(int).tolist())
    app = load_appearances(filter_player_ids=wc_ids)
    us = load_understat()

    ref_date = datetime(2026, 6, 1)  # pre-WC opening day
    print(f"\n=== Player form as of {ref_date.date()} ===\n")

    if test_mode:
        for name in ["Vinícius Júnior", "Mehdi Taremi", "Alphonso Davies", "Kylian Mbappé"]:
            form = compute_player_form(app, us, resolver, us_resolver, name, ref_date)
            print(f"{name}")
            for k, v in form.items():
                if k != "player":
                    print(f"  {k:<22} {v}")
            print()
        return

    # Full run on all WC squad players
    with open(ROOT / "frontend/src/data/squad_players.json") as f:
        squads = _json.load(f)

    rows = []
    for team, players in squads.items():
        for p in players:
            form = compute_player_form(app, us, resolver, us_resolver, p["name"], ref_date)
            form["team"] = team
            form["wc_name"] = p["name"]
            form["wc_pos"] = p.get("pos", "")
            form["wc_club"] = p.get("club", "")
            rows.append(form)

    df = pd.DataFrame(rows)
    out_path = ROOT / "data/processed/player_form_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  Total players: {len(df)}")
    print(f"  Matched in TM: {df['matched'].sum()}")
    print(f"  Has xG (Understat): {df['us_xg'].notna().sum()}")
    print(f"\nTop 5 by trailing xG/90 (min 1000 minutes):")
    qual = df[(df["us_minutes"].fillna(0) >= 1000) & df["us_xg_per_90"].notna()].copy()
    print(qual.nlargest(5, "us_xg_per_90")[["team","wc_name","us_minutes","us_xg_per_90","us_xa_per_90"]].to_string(index=False))


if __name__ == "__main__":
    main()
