"""
Build team-level aggregated features from player ratings.

Two modes:
1. 2026 WC squads: uses scraped squad data + FC26 ratings (squad_ratings_2026.csv)
2. Historical (FIFA 15→FC26): uses top-26 players by nationality per year as proxy squads

Output: data/processed/team_features_by_year.csv
  - One row per (team, year) with 22 aggregated features
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EA_DIR = DATA_DIR / "ea_fc"

# Map training data team names -> EA FC nationality names
# Covers all known mismatches between our match data and EA naming
TEAM_NAME_MAP = {
    # WC 2026 teams
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    # Other teams in training data
    "North Korea": "Korea DPR",
    "China PR": "China PR",
    "Republic of Ireland": "Republic of Ireland",
    "Northern Ireland": "Northern Ireland",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Antigua and Barbuda": "Antigua and Barbuda",
    "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    "Eswatini": "Eswatini",
    "Kyrgyzstan": "Kyrgyz Republic",
}
EA_TO_TEAM = {v: k for k, v in TEAM_NAME_MAP.items()}

# Position mapping for EA positions -> simplified
POS_MAP = {
    'GK': 'GK',
    'CB': 'DF', 'RB': 'DF', 'LB': 'DF', 'RWB': 'DF', 'LWB': 'DF',
    'CDM': 'MF', 'CM': 'MF', 'CAM': 'MF', 'RM': 'MF', 'LM': 'MF',
    'RW': 'FW', 'LW': 'FW', 'CF': 'FW', 'ST': 'FW',
}

# Imputation constants
IMPUTE_DISCOUNT = 8  # domestic league players ~8 points below European exports
IMPUTE_FLOOR = 55    # lowest realistic international player rating

# The 22 features we compute
FEATURE_COLS = [
    # Core quality
    'squad_avg_overall', 'squad_median_overall', 'squad_std_overall',
    'squad_top3_avg', 'squad_bottom5_avg',
    # Positional
    'gk_avg', 'def_avg', 'mid_avg', 'fwd_avg',
    'strongest_unit', 'weakest_unit',
    # Market value
    'squad_total_value', 'squad_avg_value',
    # Squad profile
    'squad_avg_age', 'squad_avg_potential_gap', 'squad_avg_caps',
    # Big 6 face stats
    'team_pace', 'team_shooting', 'team_passing',
    'team_dribbling', 'team_defending', 'team_physic',
]


def get_primary_position(pos_str: str) -> str:
    """Map EA multi-position string to GK/DF/MF/FW using first listed position."""
    if not isinstance(pos_str, str) or pos_str.strip() == '':
        return 'MF'  # default
    first_pos = pos_str.split(',')[0].strip()
    return POS_MAP.get(first_pos, 'MF')


def impute_missing_ratings(df: pd.DataFrame, rating_col: str = 'overall') -> pd.DataFrame:
    """
    For players with missing ratings, impute as (team_matched_avg - discount).
    This corrects the upward bias from only matching European-based players.
    """
    df = df.copy()
    for team in df['team'].unique():
        mask_team = df['team'] == team
        matched_avg = df.loc[mask_team, rating_col].mean()

        if pd.isna(matched_avg):
            # Entire team unmatched — use global median of weak teams
            imputed = IMPUTE_FLOOR
        else:
            imputed = max(IMPUTE_FLOOR, matched_avg - IMPUTE_DISCOUNT)

        # Fill NaN for this team
        mask_missing = mask_team & df[rating_col].isna()
        df.loc[mask_missing, rating_col] = imputed

    return df


def impute_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Impute all numeric rating columns, not just overall."""
    stat_cols = ['overall', 'potential', 'international_reputation',
                 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                 'value_eur', 'wage_eur']

    df = df.copy()
    for col in stat_cols:
        if col not in df.columns:
            continue
        for team in df['team'].unique():
            mask_team = df['team'] == team
            matched_avg = df.loc[mask_team, col].mean()

            if pd.isna(matched_avg):
                if col == 'value_eur':
                    imputed = 100_000
                elif col == 'wage_eur':
                    imputed = 1_000
                else:
                    imputed = IMPUTE_FLOOR
            else:
                if col in ('value_eur', 'wage_eur'):
                    # Value discount: 50% of matched avg for unmatched
                    imputed = matched_avg * 0.3
                elif col == 'international_reputation':
                    imputed = max(1, matched_avg - 1)
                else:
                    imputed = max(IMPUTE_FLOOR, matched_avg - IMPUTE_DISCOUNT)

            mask_missing = mask_team & df[col].isna()
            df.loc[mask_missing, col] = imputed

    return df


def compute_team_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 22 team-level features from player-level data.

    Input: DataFrame with columns: team, position, overall, potential, pace, shooting,
           passing, dribbling, defending, physic, value_eur, age, caps
    Output: DataFrame with one row per team and 22 feature columns
    """
    rows = []

    for team, group in players_df.groupby('team'):
        ovr = group['overall']
        row = {'team': team}

        # --- Core quality ---
        row['squad_avg_overall'] = ovr.mean()
        row['squad_median_overall'] = ovr.median()
        row['squad_std_overall'] = ovr.std() if len(ovr) > 1 else 0
        row['squad_top3_avg'] = ovr.nlargest(3).mean() if len(ovr) >= 3 else ovr.max()
        row['squad_bottom5_avg'] = ovr.nsmallest(5).mean() if len(ovr) >= 5 else ovr.min()

        # --- Positional strength ---
        # Use 'position' column (GK/DF/MF/FW)
        pos_col = 'position' if 'position' in group.columns else 'pos_group'
        pos_avgs = {}
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_players = group[group[pos_col] == pos]['overall']
            pos_avgs[pos] = pos_players.mean() if len(pos_players) > 0 else ovr.mean()

        row['gk_avg'] = pos_avgs['GK']
        row['def_avg'] = pos_avgs['DF']
        row['mid_avg'] = pos_avgs['MF']
        row['fwd_avg'] = pos_avgs['FW']
        unit_vals = [pos_avgs['GK'], pos_avgs['DF'], pos_avgs['MF'], pos_avgs['FW']]
        row['strongest_unit'] = max(unit_vals)
        row['weakest_unit'] = min(unit_vals)

        # --- Market value ---
        if 'value_eur' in group.columns:
            row['squad_total_value'] = group['value_eur'].sum()
            row['squad_avg_value'] = group['value_eur'].mean()
        else:
            row['squad_total_value'] = 0
            row['squad_avg_value'] = 0

        # --- Squad profile ---
        row['squad_avg_age'] = group['age'].mean() if 'age' in group.columns and group['age'].notna().any() else 27
        row['squad_avg_potential_gap'] = (group['potential'] - group['overall']).mean() if 'potential' in group.columns else 0
        row['squad_avg_caps'] = group['caps'].mean() if 'caps' in group.columns and group['caps'].notna().any() else 0

        # --- Big 6 face stats ---
        for stat, feat_name in [('pace', 'team_pace'), ('shooting', 'team_shooting'),
                                 ('passing', 'team_passing'), ('dribbling', 'team_dribbling'),
                                 ('defending', 'team_defending'), ('physic', 'team_physic')]:
            row[feat_name] = group[stat].mean() if stat in group.columns else 0

        rows.append(row)

    result = pd.DataFrame(rows)

    # Round numeric columns
    numeric_cols = [c for c in FEATURE_COLS if c in result.columns]
    result[numeric_cols] = result[numeric_cols].round(2)

    return result


def build_2026_features() -> pd.DataFrame:
    """Build team features from 2026 scraped squad + FC26 ratings."""
    print("Building 2026 team features from squad_ratings_2026.csv...")
    df = pd.read_csv(PROCESSED_DIR / "squad_ratings_2026.csv")
    print(f"  Loaded {len(df)} players, {df['team'].nunique()} teams")

    # Impute missing ratings
    matched_before = df['overall'].notna().sum()
    df = impute_all_stats(df)
    print(f"  Imputed {len(df) - matched_before} missing player ratings")

    # Ensure caps is numeric
    df['caps'] = pd.to_numeric(df['caps'], errors='coerce').fillna(0)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Compute features
    features = compute_team_features(df)
    features['year'] = 2026
    features['source'] = 'squad_scrape'
    return features


def process_ea_dataset(df: pd.DataFrame, year: int, source: str,
                       min_players: int = 11) -> pd.DataFrame:
    """
    Build team features for ALL nationalities from an EA dataset.
    Takes top-26 players per nationality as proxy squad.
    Skips nationalities with fewer than min_players.
    """
    year_features = []
    all_nats = df['nationality_name'].dropna().unique()

    for ea_nat in all_nats:
        nat_players = df[df['nationality_name'] == ea_nat].copy()
        if len(nat_players) < min_players:
            continue

        squad = nat_players.nlargest(min(26, len(nat_players)), 'overall').copy()
        team_name = EA_TO_TEAM.get(ea_nat, ea_nat)
        squad['team'] = team_name

        if 'player_positions' in squad.columns:
            squad['position'] = squad['player_positions'].apply(get_primary_position)
        else:
            squad['position'] = 'MF'
        squad['caps'] = 0
        squad['age'] = 27

        year_features.append(squad)

    if not year_features:
        return pd.DataFrame()

    combined = pd.concat(year_features, ignore_index=True)
    features = compute_team_features(combined)
    features['year'] = year
    features['source'] = source
    return features


def build_historical_features() -> pd.DataFrame:
    """
    Build team features for FIFA 15→FC26 using top-26 by nationality.
    Processes ALL nationalities (not just 48 WC teams).
    """
    print("\nBuilding historical team features (ALL nationalities)...")

    all_features = []

    # --- FIFA 15 to FC 24 ---
    print("  Loading FIFA 15→FC24 data...")
    cols_needed = ['fifa_version', 'long_name', 'nationality_name', 'overall', 'potential',
                   'player_positions', 'international_reputation',
                   'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                   'value_eur', 'wage_eur']
    hist_df = pd.read_csv(EA_DIR / "fifa15_to_fc24" / "male_players.csv",
                          usecols=cols_needed, low_memory=False)

    version_to_year = {v: 2000 + int(v) - 1 for v in range(15, 25)}

    for version in sorted(hist_df['fifa_version'].dropna().unique()):
        version_int = int(version)
        year = version_to_year.get(version_int)
        if year is None:
            continue

        print(f"  FIFA {version_int} (year {year})...", end=" ")
        vdf = hist_df[hist_df['fifa_version'] == version].copy()
        features = process_ea_dataset(vdf, year, f'fifa_{version_int}_top26')
        if len(features) > 0:
            all_features.append(features)
            print(f"{len(features)} teams")
        else:
            print("0 teams")

    # --- FC 25 ---
    fc25_path = EA_DIR / "fc25" / "player-data-full-2025-june.csv"
    if fc25_path.exists():
        print("  Loading FC 25...")
        fc25 = pd.read_csv(fc25_path, low_memory=False)

        fc25_rename = {
            'full_name': 'long_name',
            'name': 'short_name',
            'overall_rating': 'overall',
            'positions': 'player_positions',
            'value': 'value_eur',
            'wage': 'wage_eur',
        }
        fc25 = fc25.rename(columns={k: v for k, v in fc25_rename.items() if k in fc25.columns})

        # Map nationality via player_id from FC24/FC26
        if 'nationality_name' not in fc25.columns and 'player_id' in fc25.columns:
            fc26_nat = pd.read_csv(EA_DIR / "fc26" / "FC26_20250921.csv",
                                   usecols=['player_id', 'nationality_name'], low_memory=False)
            fc24_nat = hist_df[hist_df['fifa_version'] == 24][['long_name', 'nationality_name']].drop_duplicates()

            id_map = dict(zip(fc26_nat['player_id'], fc26_nat['nationality_name']))
            fc25['nationality_name'] = fc25['player_id'].map(id_map)

            unmapped = fc25['nationality_name'].isna()
            if unmapped.any() and 'long_name' in fc25.columns:
                fc26_name = pd.read_csv(EA_DIR / "fc26" / "FC26_20250921.csv",
                                        usecols=['long_name', 'nationality_name'], low_memory=False)
                name_map_df = pd.concat([fc26_name, fc24_nat]).drop_duplicates(subset='long_name', keep='first')
                name_dict = dict(zip(name_map_df['long_name'], name_map_df['nationality_name']))
                fc25.loc[unmapped, 'nationality_name'] = fc25.loc[unmapped, 'long_name'].map(name_dict)

            mapped = fc25['nationality_name'].notna().sum()
            print(f"    Mapped nationality for {mapped}/{len(fc25)} FC25 players")

        # Parse value/wage strings
        for col in ['value_eur', 'wage_eur']:
            if col in fc25.columns and fc25[col].dtype == object:
                def parse_value(v):
                    if not isinstance(v, str):
                        return 0
                    v = v.replace('€', '').replace(',', '').strip()
                    if 'M' in v:
                        return float(v.replace('M', '')) * 1_000_000
                    elif 'K' in v:
                        return float(v.replace('K', '')) * 1_000
                    try:
                        return float(v)
                    except ValueError:
                        return 0
                fc25[col] = fc25[col].apply(parse_value)

        # Build face stats from detailed stats if missing
        if 'pace' not in fc25.columns and 'acceleration' in fc25.columns:
            fc25['pace'] = (fc25.get('acceleration', 0) + fc25.get('sprint_speed', 0)) / 2
            fc25['shooting'] = fc25[['finishing', 'shot_power', 'long_shots', 'volleys', 'positioning']].mean(axis=1) if 'finishing' in fc25.columns else 0
            fc25['passing'] = (fc25.get('short_passing', 0) + fc25.get('long_passing', 0) + fc25.get('vision', 0)) / 3
            fc25['dribbling'] = (fc25.get('dribbling', fc25.get('ball_control', 0)))
            fc25['defending'] = (fc25.get('defensive_awareness', 0) + fc25.get('standing_tackle', 0) + fc25.get('sliding_tackle', 0)) / 3
            fc25['physic'] = (fc25.get('strength', 0) + fc25.get('stamina', 0) + fc25.get('aggression', 0)) / 3

        features = process_ea_dataset(fc25, 2024, 'fc25_top26')
        if len(features) > 0:
            all_features.append(features)
            print(f"    {len(features)} teams")

    # --- FC 26 ---
    print("  Loading FC 26 (top-26 proxy)...")
    fc26 = pd.read_csv(EA_DIR / "fc26" / "FC26_20250921.csv", low_memory=False)
    features = process_ea_dataset(fc26, 2025, 'fc26_top26')
    if len(features) > 0:
        all_features.append(features)
        print(f"    {len(features)} teams")

    return pd.concat(all_features, ignore_index=True)


def main():
    # Build 2026 from actual scraped squads
    features_2026 = build_2026_features()

    # Build historical from top-26 proxy
    features_hist = build_historical_features()

    # Combine
    # For 2026, prefer the scraped squad version over top-26 proxy
    all_features = pd.concat([features_hist, features_2026], ignore_index=True)

    # Sort
    all_features = all_features.sort_values(['year', 'team']).reset_index(drop=True)

    # Save
    output_path = PROCESSED_DIR / "team_features_by_year.csv"
    all_features.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Shape: {all_features.shape}")
    print(f"Years: {sorted(all_features['year'].unique())}")
    print(f"Teams per year:")
    print(all_features.groupby('year')['team'].count().to_string())

    # Sanity check: show top and bottom teams for 2026
    print(f"\n--- 2026 Team Rankings (by squad_avg_overall) ---")
    t26 = all_features[all_features['year'] == 2026].sort_values('squad_avg_overall', ascending=False)
    for _, r in t26.iterrows():
        print(f"  {r['team']:25s} | avg: {r['squad_avg_overall']:5.1f} | top3: {r['squad_top3_avg']:5.1f} | value: {r['squad_total_value']:>14,.0f} | src: {r['source']}")


if __name__ == "__main__":
    main()
