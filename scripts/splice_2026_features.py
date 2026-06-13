"""
Splice the freshly-computed 2026 team features into team_features_by_year.csv.

Strategy: only replace the 2026 row for teams with high enough EA FC 26 coverage.
For low-coverage teams, the original FM23-imputed row is kept so we don't introduce
bias from small-sample squad averages (e.g., Iran with 3/26 rated players).

Output: team_features_by_year.csv (overwritten — original backed up alongside).
"""

import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TFBY = ROOT / "data/processed/team_features_by_year.csv"
NEW_2026 = ROOT / "data/processed/team_features_2026.csv"
BACKUP = ROOT / "data/processed/team_features_by_year.backup.csv"

COVERAGE_THRESHOLD = 0.80  # only replace if ≥80% of squad has ratings


def main():
    # Backup once
    if not BACKUP.exists():
        shutil.copy(TFBY, BACKUP)
        print(f"Backed up {TFBY} → {BACKUP}")

    df = pd.read_csv(TFBY)
    new = pd.read_csv(NEW_2026)
    new = new.set_index("team")

    # Determine which teams to replace
    old_2026_teams = set(df[df["year"] == 2026]["team"])
    replace_mask = new["coverage"] >= COVERAGE_THRESHOLD
    replace_teams = set(new[replace_mask].index)
    # Also include teams not in old data at all — they must come from new
    missing_from_old = set(new.index) - old_2026_teams
    replace_teams |= missing_from_old
    skip_teams = set(new.index) - replace_teams

    print(f"\nCoverage threshold: {COVERAGE_THRESHOLD*100:.0f}%")
    print(f"Teams to replace (high-coverage or missing from old): {len(replace_teams)}")
    print(f"  ... of which forced-replace (missing from old): {sorted(missing_from_old)}")
    print(f"Teams to keep old row (low-coverage): {len(skip_teams)}")
    print(f"\nKept-old: {sorted(skip_teams)}")

    # Build new 2026 rows in the same column order as team_features_by_year.csv
    target_cols = [c for c in df.columns if c not in ("year", "source", "team")]
    new_rows = []
    for team in sorted(new.index):
        if team not in replace_teams:
            continue
        row = {"team": team}
        for c in target_cols:
            row[c] = new.loc[team, c] if c in new.columns else None
        row["year"] = 2026
        row["source"] = "2026_real_squads"
        new_rows.append(row)

    # Add a row for Curaçao if it's in replace_teams (was missing from old data)
    # (already handled above)

    # Drop existing 2026 rows for replaced teams; keep low-coverage 2026 rows as-is
    keep_old_2026 = df[(df["year"] == 2026) & (df["team"].isin(skip_teams))]
    not_2026 = df[df["year"] != 2026]

    new_2026_df = pd.DataFrame(new_rows, columns=df.columns)
    final = pd.concat([not_2026, keep_old_2026, new_2026_df], ignore_index=True)
    final = final.sort_values(["year", "team"]).reset_index(drop=True)

    final.to_csv(TFBY, index=False)
    final_2026 = final[final["year"] == 2026]
    print(f"\nWrote {TFBY}")
    print(f"  Total rows: {len(final)}")
    print(f"  2026 rows: {len(final_2026)}")
    print(f"  2026 from new real squads: {(final_2026['source']=='2026_real_squads').sum()}")
    print(f"  2026 from original (kept due to low coverage): {(final_2026['source']!='2026_real_squads').sum()}")


if __name__ == "__main__":
    main()
