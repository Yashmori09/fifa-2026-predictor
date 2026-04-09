"""
Build rich squad + EA FC ratings dataset for all 48 WC 2026 teams.
Merges Wikipedia squad data with FC 26 player ratings.

Output: data/processed/squad_ratings_2026.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EA_DIR = DATA_DIR / "ea_fc"

# Team name mapping: our names -> EA FC names
TEAM_NAME_MAP = {
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
}

# Reverse map for display
TEAM_NAME_MAP_REV = {v: k for k, v in TEAM_NAME_MAP.items()}


def strip_diacritics(s: str) -> str:
    """Remove accents/diacritics from a string."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if not isinstance(name, str):
        return ""
    name = strip_diacritics(name).strip().lower()
    name = re.sub(r'[^\w\s]', '', name)  # remove punctuation
    name = re.sub(r'\s+', ' ', name)
    return name


def fuzzy_match_score(a: str, b: str) -> float:
    """Fuzzy match score between two names."""
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def match_players(squad_df: pd.DataFrame, fc_df: pd.DataFrame) -> pd.DataFrame:
    """Match squad players to EA FC ratings by name within same nationality."""

    matched = []
    unmatched = []

    for _, player in squad_df.iterrows():
        team = player['team']
        fc_nationality = TEAM_NAME_MAP.get(team, team)
        name = player['name']

        # Filter FC players by nationality
        fc_team = fc_df[fc_df['nationality_name'] == fc_nationality].copy()

        if len(fc_team) == 0:
            unmatched.append({'team': team, 'name': name, 'reason': 'no FC players for nationality'})
            matched.append({**player.to_dict(), **{c: None for c in FC_COLS}})
            continue

        name_norm = normalize_name(name)
        parts = name.split()
        last_name = parts[-1].lower() if parts else ""
        first_name = parts[0].lower() if parts else ""
        last_norm = strip_diacritics(last_name).lower()
        first_norm = strip_diacritics(first_name).lower()

        # Add normalized columns for matching
        fc_team = fc_team.copy()
        fc_team['_long_norm'] = fc_team['long_name'].apply(lambda x: normalize_name(x) if isinstance(x, str) else '')
        fc_team['_short_norm'] = fc_team['short_name'].apply(lambda x: normalize_name(x) if isinstance(x, str) else '')

        # Try exact match (normalized)
        exact = fc_team[
            (fc_team['_long_norm'] == name_norm) |
            (fc_team['_short_norm'] == name_norm)
        ]

        if len(exact) >= 1:
            best = exact.nlargest(1, 'overall').iloc[0]
            matched.append({**player.to_dict(), **{c: best.get(c) for c in FC_COLS}})
            continue

        # Try partial match — last name (normalized)
        partial = fc_team[
            fc_team['_long_norm'].str.contains(last_norm, na=False, regex=False) |
            fc_team['_short_norm'].str.contains(last_norm, na=False, regex=False)
        ]

        if len(partial) == 1:
            best = partial.iloc[0]
            matched.append({**player.to_dict(), **{c: best.get(c) for c in FC_COLS}})
            continue
        elif len(partial) > 1:
            # Multiple matches — try adding first name
            refined = partial[
                partial['_long_norm'].str.contains(first_norm, na=False, regex=False)
            ]
            if len(refined) >= 1:
                best = refined.nlargest(1, 'overall').iloc[0]
                matched.append({**player.to_dict(), **{c: best.get(c) for c in FC_COLS}})
                continue
            else:
                best = partial.nlargest(1, 'overall').iloc[0]
                matched.append({**player.to_dict(), **{c: best.get(c) for c in FC_COLS}})
                continue

        # Fuzzy match as last resort — try both long_name and short_name
        fc_team['_score_long'] = fc_team['_long_norm'].apply(lambda x: SequenceMatcher(None, name_norm, x).ratio())
        fc_team['_score_short'] = fc_team['_short_norm'].apply(lambda x: SequenceMatcher(None, name_norm, x).ratio())
        fc_team['_score'] = fc_team[['_score_long', '_score_short']].max(axis=1)
        best_score = fc_team['_score'].max()

        if best_score >= 0.55:
            best = fc_team.nlargest(1, '_score').iloc[0]
            matched.append({**player.to_dict(), **{c: best.get(c) for c in FC_COLS}})
            continue

        # No match found
        unmatched.append({'team': team, 'name': name, 'reason': f'best fuzzy score {best_score:.2f}'})
        matched.append({**player.to_dict(), **{c: None for c in FC_COLS}})

    return pd.DataFrame(matched), pd.DataFrame(unmatched)


# Columns to pull from EA FC
FC_COLS = [
    # Identity
    'player_id', 'short_name', 'long_name', 'player_positions',
    # Ratings
    'overall', 'potential', 'international_reputation',
    # Face stats (the big 6)
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    # Value
    'value_eur', 'wage_eur',
    # Physical
    'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot', 'skill_moves',
    # Club info from EA
    'club_name', 'league_name', 'league_level',
    # Attacking detail
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys',
    # Skill detail
    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
    'skill_long_passing', 'skill_ball_control',
    # Movement
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance',
    # Power
    'power_shot_power', 'power_jumping', 'power_stamina',
    'power_strength', 'power_long_shots',
    # Mentality
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
    'mentality_vision', 'mentality_penalties', 'mentality_composure',
    # Defending detail
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    # GK
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
    'goalkeeping_positioning', 'goalkeeping_reflexes',
]


def main():
    print("Loading data...")
    squads = pd.read_csv(PROCESSED_DIR / "squads_2026.csv")
    fc26 = pd.read_csv(EA_DIR / "fc26" / "FC26_20250921.csv", low_memory=False)

    print(f"Squad players: {len(squads)}")
    print(f"FC 26 players: {len(fc26)}")

    # Rename EA columns that conflict with squad columns
    # squad has: name, position, dob, age, caps, goals, club, club_country, jersey_number
    # EA has: player_positions, dob, age, height_cm, weight_kg, club_name
    # We'll prefix EA cols to avoid confusion, except the ones we specifically want

    print("\nMatching players...")
    matched_df, unmatched_df = match_players(squads, fc26)

    # Rename EA columns to have ea_ prefix where they conflict
    rename_map = {
        'player_id': 'ea_player_id',
        'short_name': 'ea_short_name',
        'long_name': 'ea_long_name',
        'player_positions': 'ea_positions',
        'height_cm': 'ea_height_cm',
        'weight_kg': 'ea_weight_kg',
        'club_name': 'ea_club',
        'league_name': 'ea_league',
        'league_level': 'ea_league_level',
    }
    matched_df = matched_df.rename(columns=rename_map)

    # Stats
    total = len(matched_df)
    with_rating = matched_df['overall'].notna().sum()
    without_rating = matched_df['overall'].isna().sum()

    print(f"\nResults:")
    print(f"  Total players: {total}")
    print(f"  Matched with EA rating: {with_rating} ({with_rating/total*100:.1f}%)")
    print(f"  Unmatched: {without_rating} ({without_rating/total*100:.1f}%)")

    if len(unmatched_df) > 0:
        print(f"\nUnmatched players ({len(unmatched_df)}):")
        for _, row in unmatched_df.iterrows():
            print(f"  {row['team']:25s} | {row['name']:30s} | {row['reason']}")

    # Per-team coverage
    print(f"\nPer-team coverage:")
    coverage = matched_df.groupby('team').agg(
        squad_size=('name', 'count'),
        matched=('overall', lambda x: x.notna().sum()),
        avg_overall=('overall', 'mean'),
        top_rating=('overall', 'max'),
    ).reset_index()
    coverage['match_pct'] = (coverage['matched'] / coverage['squad_size'] * 100).round(1)
    coverage = coverage.sort_values('avg_overall', ascending=False)
    print(coverage.to_string(index=False))

    # Save
    output_path = PROCESSED_DIR / "squad_ratings_2026.csv"
    matched_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {matched_df.shape}")


if __name__ == "__main__":
    main()
