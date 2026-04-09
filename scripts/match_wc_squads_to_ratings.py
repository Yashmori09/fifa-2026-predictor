"""
Match actual WC squad players (2014, 2018, 2022) to their corresponding
FIFA year ratings, compute team features, and update team_features_by_year.csv.

WC 2014 -> FIFA 15 (version 15, most accurate for that tournament)
WC 2018 -> FIFA 19 (version 19)
WC 2022 -> FIFA 23 (version 23)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

# Reuse from build_squad_ratings.py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EA_DIR = DATA_DIR / "ea_fc"

# Import team feature computation from build_team_features
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_team_features import (
    compute_team_features, impute_all_stats,
    TEAM_NAME_MAP, EA_TO_TEAM, POS_MAP, get_primary_position
)

# WC year -> FIFA version mapping
WC_FIFA_MAP = {
    2014: 15,  # FIFA 15 ratings closest to 2014 WC squads
    2018: 19,  # FIFA 19
    2022: 23,  # FIFA 23
}


def strip_diacritics(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = strip_diacritics(name).strip().lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def match_squad_to_ratings(squad_df: pd.DataFrame, fc_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Match WC squad players to FIFA ratings by name + nationality."""

    fc_cols = ['overall', 'potential', 'international_reputation',
               'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
               'value_eur', 'wage_eur', 'player_positions', 'long_name', 'short_name']

    matched_rows = []
    match_count = 0
    total = 0

    for _, player in squad_df.iterrows():
        team = player['team']
        ea_nat = TEAM_NAME_MAP.get(team, team)
        name = player['name']
        total += 1

        fc_team = fc_df[fc_df['nationality_name'] == ea_nat].copy()
        if len(fc_team) == 0:
            matched_rows.append({**player.to_dict(), **{c: None for c in fc_cols}})
            continue

        name_norm = normalize_name(name)
        parts = name.split()
        last_name = parts[-1].lower() if parts else ""
        first_name = parts[0].lower() if parts else ""
        last_norm = strip_diacritics(last_name).lower()
        first_norm = strip_diacritics(first_name).lower()

        fc_team['_long_norm'] = fc_team['long_name'].apply(lambda x: normalize_name(x) if isinstance(x, str) else '')
        fc_team['_short_norm'] = fc_team['short_name'].apply(lambda x: normalize_name(x) if isinstance(x, str) else '')

        # Exact match
        exact = fc_team[
            (fc_team['_long_norm'] == name_norm) |
            (fc_team['_short_norm'] == name_norm)
        ]
        if len(exact) >= 1:
            best = exact.nlargest(1, 'overall').iloc[0]
            matched_rows.append({**player.to_dict(), **{c: best.get(c) for c in fc_cols}})
            match_count += 1
            continue

        # Partial last name
        partial = fc_team[
            fc_team['_long_norm'].str.contains(last_norm, na=False, regex=False) |
            fc_team['_short_norm'].str.contains(last_norm, na=False, regex=False)
        ]
        if len(partial) == 1:
            best = partial.iloc[0]
            matched_rows.append({**player.to_dict(), **{c: best.get(c) for c in fc_cols}})
            match_count += 1
            continue
        elif len(partial) > 1:
            refined = partial[partial['_long_norm'].str.contains(first_norm, na=False, regex=False)]
            if len(refined) >= 1:
                best = refined.nlargest(1, 'overall').iloc[0]
            else:
                best = partial.nlargest(1, 'overall').iloc[0]
            matched_rows.append({**player.to_dict(), **{c: best.get(c) for c in fc_cols}})
            match_count += 1
            continue

        # Fuzzy match
        fc_team['_score_long'] = fc_team['_long_norm'].apply(lambda x: SequenceMatcher(None, name_norm, x).ratio())
        fc_team['_score_short'] = fc_team['_short_norm'].apply(lambda x: SequenceMatcher(None, name_norm, x).ratio())
        fc_team['_score'] = fc_team[['_score_long', '_score_short']].max(axis=1)
        best_score = fc_team['_score'].max()

        if best_score >= 0.55:
            best = fc_team.nlargest(1, '_score').iloc[0]
            matched_rows.append({**player.to_dict(), **{c: best.get(c) for c in fc_cols}})
            match_count += 1
            continue

        matched_rows.append({**player.to_dict(), **{c: None for c in fc_cols}})

    print(f"    Matched: {match_count}/{total} ({match_count/total*100:.1f}%)")
    return pd.DataFrame(matched_rows)


def main():
    print("Loading historical FIFA data...")
    hist_df = pd.read_csv(EA_DIR / "fifa15_to_fc24" / "male_players.csv", low_memory=False)

    all_wc_features = []

    for wc_year, fifa_version in WC_FIFA_MAP.items():
        print(f"\n{'='*60}")
        print(f"Processing {wc_year} World Cup (using FIFA {fifa_version} ratings)")

        # Load WC squad
        squad_path = PROCESSED_DIR / f"wc_squads_{wc_year}.csv"
        squad_df = pd.read_csv(squad_path)
        print(f"  Squad: {len(squad_df)} players, {squad_df['team'].nunique()} teams")

        # Get corresponding FIFA version
        fc_df = hist_df[hist_df['fifa_version'] == fifa_version].copy()
        print(f"  FIFA {fifa_version}: {len(fc_df)} players")

        # Match
        print("  Matching players...")
        matched = match_squad_to_ratings(squad_df, fc_df, wc_year)

        # Impute missing
        before = matched['overall'].notna().sum()
        matched = impute_all_stats(matched)
        imputed = len(matched) - before
        print(f"  Imputed {imputed} missing ratings")

        # Ensure numeric
        matched['caps'] = pd.to_numeric(matched['caps'], errors='coerce').fillna(0)
        matched['age'] = 27  # placeholder since WC squad DOB format varies

        # Compute team features
        features = compute_team_features(matched)

        # Map year: FIFA 15 -> year 2014 (WC year)
        features['year'] = wc_year
        features['source'] = f'wc_{wc_year}_actual'

        all_wc_features.append(features)

        # Show top teams
        top = features.sort_values('squad_avg_overall', ascending=False).head(10)
        print(f"\n  Top 10 teams by squad avg:")
        for _, r in top.iterrows():
            print(f"    {r['team']:25s} | avg: {r['squad_avg_overall']:5.1f} | top3: {r['squad_top3_avg']:5.1f}")

    wc_features = pd.concat(all_wc_features, ignore_index=True)

    # Load existing team features and replace WC years with actual squad data
    existing = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")
    print(f"\nExisting features: {existing.shape}")

    # For WC years, replace proxy data with actual squad data (only for teams that were in the WC)
    for wc_year in WC_FIFA_MAP.keys():
        wc_teams = wc_features[wc_features['year'] == wc_year]['team'].tolist()
        # Remove proxy rows for WC teams in that year
        mask = (existing['year'] == wc_year) & (existing['team'].isin(wc_teams))
        existing = existing[~mask]
        print(f"  Replaced {mask.sum()} proxy rows for {wc_year} WC teams")

    # Add actual WC features
    updated = pd.concat([existing, wc_features], ignore_index=True)
    updated = updated.sort_values(['year', 'team']).reset_index(drop=True)

    # Save
    output_path = PROCESSED_DIR / "team_features_by_year.csv"
    updated.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved: {output_path}")
    print(f"Shape: {updated.shape}")
    print(f"Teams per year:")
    print(updated.groupby('year')['team'].count().to_string())

    # Also save the matched player-level data for reference
    for wc_year in WC_FIFA_MAP.keys():
        squad_path = PROCESSED_DIR / f"wc_squads_{wc_year}.csv"
        # Already saved by scraper, no need to re-save


if __name__ == "__main__":
    main()
