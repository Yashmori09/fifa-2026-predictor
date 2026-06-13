"""
Build frontend/src/data/squad_players.json from squad_ratings_2026.csv.

For each player:
  - If overall is present → use those ratings (from FC 26 or FM 23).
  - Else → impute from team's position-group average. If team has no rated
    players at that position, fall back to team's overall avg minus a discount.

Output: frontend/src/data/squad_players.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQUAD = ROOT / "data/processed/squad_ratings_2026.csv"
OUT = ROOT / "frontend/src/data/squad_players.json"
BACKUP = ROOT / "frontend/src/data/squad_players.backup.json"

# CSV column → JSON key
MAP = {
    "name": "name", "position": "pos", "age": "age", "caps": "caps", "goals": "goals",
    "club": "club",
    "overall": "ovr", "potential": "pot", "international_reputation": "rep", "value_eur": "val",
    "pace": "pac", "shooting": "sho", "passing": "pas",
    "dribbling": "dri", "defending": "defe", "physic": "phy",
    "goalkeeping_diving": "div", "goalkeeping_handling": "han",
    "goalkeeping_kicking": "kic", "goalkeeping_positioning": "gkp",
    "goalkeeping_reflexes": "ref",
}

OUTFIELD_KEYS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
GK_KEYS = ["goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
           "goalkeeping_positioning", "goalkeeping_reflexes"]


def rep_from_ovr(ovr):
    if pd.isna(ovr): return None
    if ovr >= 88: return 5
    if ovr >= 84: return 4
    if ovr >= 79: return 3
    if ovr >= 73: return 2
    return 1


def val_from_ovr_age(ovr, age):
    if pd.isna(ovr): return None
    base = {90: 90_000_000, 85: 50_000_000, 80: 22_000_000,
            75: 8_000_000, 70: 2_500_000, 65: 800_000, 60: 200_000}
    val = 500_000
    for t in sorted(base.keys(), reverse=True):
        if ovr >= t:
            val = base[t]; break
    if not pd.isna(age):
        if age >= 33: val = int(val * 0.4)
        elif age >= 30: val = int(val * 0.6)
        elif age >= 28: val = int(val * 0.8)
        elif age <= 20: val = int(val * 1.2)
    return val


def impute_team_pos(df, team, pos):
    """Impute ratings for a player on `team` at position `pos` from team-position averages."""
    team_df = df[(df["team"] == team) & (df["overall"].notna())]
    pos_df = team_df[team_df["position"] == pos]
    src = pos_df if len(pos_df) > 0 else team_df
    if len(src) == 0:
        return None  # no team data at all
    out = {"overall": int(round(src["overall"].mean() - 4))}  # bottom-of-team discount
    if pos == "GK":
        for col in GK_KEYS:
            vals = src[col].dropna()
            if len(vals) > 0:
                out[col] = int(round(vals.mean()))
    else:
        for col in OUTFIELD_KEYS:
            vals = src[col].dropna()
            if len(vals) > 0:
                out[col] = int(round(vals.mean()))
    return out


def main():
    df = pd.read_csv(SQUAD, low_memory=False)
    print(f"Loaded {len(df)} players")

    # Backup
    if OUT.exists() and not BACKUP.exists():
        BACKUP.write_text(OUT.read_text())
        print(f"Backed up old JSON")

    out_data = {}
    imputed_count = 0

    for _, row in df.iterrows():
        team = row["team"]
        pos = row["position"]
        player = {}
        # Copy fields that are present
        if pd.notna(row.get("name")): player["name"] = row["name"]
        if pd.notna(row.get("position")): player["pos"] = row["position"]
        if pd.notna(row.get("age")):
            try: player["age"] = int(row["age"])
            except: pass
        if pd.notna(row.get("caps")):
            try: player["caps"] = int(row["caps"])
            except: pass
        if pd.notna(row.get("goals")):
            try: player["goals"] = int(row["goals"])
            except: pass
        if pd.notna(row.get("club")): player["club"] = row["club"]

        # Ratings
        if pd.notna(row.get("overall")):
            player["ovr"] = int(row["overall"])
            if pd.notna(row.get("potential")):
                player["pot"] = int(row["potential"])
            else:
                player["pot"] = player["ovr"]
            if pd.notna(row.get("international_reputation")):
                player["rep"] = int(row["international_reputation"])
            else:
                player["rep"] = rep_from_ovr(player["ovr"])
            if pd.notna(row.get("value_eur")):
                try: player["val"] = int(row["value_eur"])
                except: player["val"] = val_from_ovr_age(player["ovr"], row.get("age"))
            else:
                player["val"] = val_from_ovr_age(player["ovr"], row.get("age"))
            # Attrs
            if pos == "GK":
                for csv_k in GK_KEYS:
                    if pd.notna(row.get(csv_k)):
                        player[MAP[csv_k]] = int(row[csv_k])
            else:
                for csv_k in OUTFIELD_KEYS:
                    if pd.notna(row.get(csv_k)):
                        player[MAP[csv_k]] = int(row[csv_k])
        else:
            # Impute from team-position average
            imp = impute_team_pos(df, team, pos)
            if imp:
                player["ovr"] = imp["overall"]
                player["pot"] = imp["overall"]
                player["rep"] = rep_from_ovr(imp["overall"])
                player["val"] = val_from_ovr_age(imp["overall"], row.get("age"))
                for csv_k, jk in MAP.items():
                    if csv_k in imp:
                        player[jk] = imp[csv_k]
                player["imputed"] = True
                imputed_count += 1

        out_data.setdefault(team, []).append(player)

    # Sort each team by overall desc
    for team in out_data:
        out_data[team].sort(key=lambda p: -(p.get("ovr") or 0))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, separators=(",", ":"))

    total_players = sum(len(v) for v in out_data.values())
    print(f"\nWrote {OUT}")
    print(f"Teams: {len(out_data)}  Players: {total_players}  Imputed: {imputed_count}")


if __name__ == "__main__":
    main()
