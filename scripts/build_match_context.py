"""
B4 — Per-match context features.

For each scheduled 2026 WC match, compute (per team in the match):
  is_host_at_home  — binary, 1 if team is co-host playing in their country
  days_rest        — days since this team's previous match in the schedule
  travel_km        — Haversine distance from previous match's city to this one

Output: data/processed/match_context_features.csv
Keyed by (match_id, team)
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHED = ROOT / "data/processed/schedule_2026.csv"
OUT = ROOT / "data/processed/match_context_features.csv"

HOSTS = {"Mexico", "Canada", "United States"}

# Host nation → cities in their country
HOST_CITIES = {
    "Mexico": {"Mexico City", "Guadalajara", "Monterrey"},
    "Canada": {"Toronto", "Vancouver"},
    "United States": {"Atlanta", "Boston", "Dallas", "Houston", "Kansas City",
                      "Los Angeles", "Miami", "New York", "Philadelphia",
                      "San Francisco", "Seattle"},
}

# Venue city → (lat, lon) for all 16 WC host cities
CITY_COORDS = {
    "Atlanta":       (33.7490, -84.3880),
    "Boston":        (42.3601, -71.0589),
    "Dallas":        (32.7767, -96.7970),
    "Guadalajara":   (20.6736, -103.3440),
    "Houston":       (29.7604, -95.3698),
    "Kansas City":   (39.0997, -94.5786),
    "Los Angeles":   (34.0522, -118.2437),
    "Mexico City":   (19.4326, -99.1332),
    "Miami":         (25.7617, -80.1918),
    "Monterrey":     (25.6866, -100.3161),
    "New York":      (40.7128, -74.0060),
    "Philadelphia":  (39.9526, -75.1652),
    "San Francisco": (37.7749, -122.4194),
    "Seattle":       (47.6062, -122.3321),
    "Toronto":       (43.6532, -79.3832),
    "Vancouver":     (49.2827, -123.1207),
}


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    aa = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(aa))


def main():
    sched = pd.read_csv(SCHED, parse_dates=["date"])
    print(f"Loaded {len(sched)} scheduled matches")

    # Build per-(match, team) row
    rows = []
    for _, m in sched.iterrows():
        for side, team in [("home", m["home_team"]), ("away", m["away_team"])]:
            rows.append({
                "match_id": m["match_id"],
                "date": m["date"],
                "team": team,
                "side": side,
                "venue": m["venue"],
                "city": m["city"],
            })
    flat = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    # Compute days_rest + travel_km per team
    def per_team(g):
        g = g.sort_values("date").reset_index(drop=True)
        g["days_rest"] = (g["date"] - g["date"].shift(1)).dt.days
        # Travel km from previous match's city
        coords = g["city"].map(CITY_COORDS)
        prev = coords.shift(1)
        g["travel_km"] = [
            round(haversine_km(p[0], p[1], c[0], c[1]), 1)
            if isinstance(c, tuple) and isinstance(p, tuple) else None
            for c, p in zip(coords, prev)
        ]
        return g

    flat = flat.groupby("team", group_keys=False).apply(per_team)

    # Host flag
    flat["is_host"] = flat["team"].isin(HOSTS).astype(int)
    flat["is_host_at_home"] = flat.apply(
        lambda r: int(r["is_host"] and r["city"] in HOST_CITIES.get(r["team"], set())),
        axis=1,
    )

    # First match: days_rest/travel_km are null
    # For prediction, treat first WC match as "fresh" (rested)
    # Fill: days_rest with a reasonable default (7), travel_km with 0
    flat["days_rest"] = flat["days_rest"].fillna(7).astype(int)
    flat["travel_km"] = flat["travel_km"].fillna(0.0)

    flat = flat[["match_id","date","team","side","city","is_host","is_host_at_home","days_rest","travel_km"]]
    flat.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(flat)} (match,team) rows\n")

    # Sanity checks
    print("Host vs non-host distribution:")
    print(flat.groupby(["is_host","is_host_at_home"]).size().to_string())
    print("\nSample host advantages:")
    hosts = flat[flat["is_host_at_home"] == 1]
    print(hosts.head(6).to_string(index=False))
    print(f"\nTotal host_at_home matches: {len(hosts)} (each host has 3 group matches)")
    print("\nTravel km stats:")
    print(flat["travel_km"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
