"""
Build team features from Football Manager 2023 data.

FM23 covers 209 nationalities (vs EA's ~90), giving us squad features
for ~46 additional teams that EA doesn't cover.

FM uses 1-20 attribute scale. We compute a composite 'overall' rating
per player, calibrate it to EA's 1-99 scale using overlapping teams,
then run the same team feature aggregation pipeline.

Output: updates data/processed/team_features_by_year.csv with FM-sourced rows.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EA_DIR = DATA_DIR / "ea_fc"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_team_features import compute_team_features, FEATURE_COLS

# ── FM country code -> match data team name ──
FM_CODE_MAP = {
    'AFG': 'Afghanistan', 'ALB': 'Albania', 'ALG': 'Algeria', 'AND': 'Andorra',
    'ANG': 'Angola', 'ARG': 'Argentina', 'ARM': 'Armenia', 'ARU': 'Aruba',
    'ASA': 'American Samoa', 'ATG': 'Antigua and Barbuda', 'AUS': 'Australia',
    'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BAH': 'Bahamas', 'BAN': 'Bangladesh',
    'BDI': 'Burundi', 'BEL': 'Belgium', 'BEN': 'Benin', 'BER': 'Bermuda',
    'BFA': 'Burkina Faso', 'BHR': 'Bahrain', 'BIH': 'Bosnia and Herzegovina',
    'BLR': 'Belarus', 'BLZ': 'Belize', 'BOL': 'Bolivia', 'BOT': 'Botswana',
    'BRA': 'Brazil', 'BRB': 'Barbados', 'BRU': 'Brunei', 'BUL': 'Bulgaria',
    'CAN': 'Canada', 'CGO': 'Congo', 'CHA': 'Chad', 'CHI': 'Chile',
    'CHN': 'China PR', 'CIV': 'Ivory Coast', 'CMR': 'Cameroon',
    'COD': 'DR Congo', 'COL': 'Colombia', 'COM': 'Comoros',
    'CPV': 'Cape Verde', 'CRC': 'Costa Rica', 'CRO': 'Croatia',
    'CTA': 'Central African Republic', 'CUB': 'Cuba', 'CUW': 'Curacao',
    'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DEN': 'Denmark',
    'DOM': 'Dominican Republic', 'ECU': 'Ecuador', 'EGY': 'Egypt',
    'ENG': 'England', 'EQG': 'Equatorial Guinea', 'ERI': 'Eritrea',
    'ESP': 'Spain', 'EST': 'Estonia', 'ETH': 'Ethiopia',
    'FIN': 'Finland', 'FRA': 'France', 'FRO': 'Faroe Islands',
    'GAB': 'Gabon', 'GAM': 'Gambia', 'GEO': 'Georgia', 'GER': 'Germany',
    'GHA': 'Ghana', 'GIB': 'Gibraltar', 'GLP': 'Guadeloupe',
    'GNB': 'Guinea-Bissau', 'GRE': 'Greece', 'GRN': 'Grenada',
    'GUA': 'Guatemala', 'GUF': 'French Guiana', 'GUI': 'Guinea',
    'GUM': 'Guam', 'GUY': 'Guyana', 'HAI': 'Haiti',
    'HKG': 'Hong Kong', 'HON': 'Honduras', 'HUN': 'Hungary',
    'IDN': 'Indonesia', 'IRL': 'Republic of Ireland', 'IRN': 'Iran',
    'IRQ': 'Iraq', 'ISL': 'Iceland', 'ISR': 'Israel', 'ITA': 'Italy',
    'JAM': 'Jamaica', 'JOR': 'Jordan', 'JPN': 'Japan',
    'KAZ': 'Kazakhstan', 'KEN': 'Kenya', 'KOR': 'South Korea',
    'KOS': 'Kosovo', 'KSA': 'Saudi Arabia', 'KUW': 'Kuwait',
    'LBR': 'Liberia', 'LBY': 'Libya', 'LCA': 'Saint Lucia',
    'LIB': 'Lebanon', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
    'LVA': 'Latvia', 'MAD': 'Madagascar', 'MAR': 'Morocco',
    'MAS': 'Malaysia', 'MDA': 'Moldova', 'MEX': 'Mexico',
    'MKD': 'North Macedonia', 'MLI': 'Mali', 'MLT': 'Malta',
    'MNE': 'Montenegro', 'MOZ': 'Mozambique', 'MRI': 'Mauritius',
    'MSR': 'Montserrat', 'MTN': 'Mauritania', 'MTQ': 'Martinique',
    'NCL': 'New Caledonia', 'NED': 'Netherlands', 'NGA': 'Nigeria',
    'NIG': 'Niger', 'NIR': 'Northern Ireland', 'NOR': 'Norway',
    'NZL': 'New Zealand', 'OMA': 'Oman', 'PAK': 'Pakistan',
    'PAL': 'Palestine', 'PAN': 'Panama', 'PAR': 'Paraguay',
    'PER': 'Peru', 'PHI': 'Philippines', 'POL': 'Poland',
    'POR': 'Portugal', 'PRK': 'North Korea', 'PUR': 'Puerto Rico',
    'QAT': 'Qatar', 'ROU': 'Romania', 'RSA': 'South Africa',
    'RUS': 'Russia', 'RWA': 'Rwanda', 'SCO': 'Scotland',
    'SEN': 'Senegal', 'SKN': 'Saint Kitts and Nevis',
    'SLE': 'Sierra Leone', 'SLV': 'El Salvador', 'SMA': 'Saint Martin',
    'SMR': 'San Marino', 'SOM': 'Somalia', 'SRB': 'Serbia',
    'SRI': 'Sri Lanka', 'SSD': 'South Sudan', 'STP': 'Sao Tome and Principe',
    'SUI': 'Switzerland', 'SUR': 'Suriname', 'SVK': 'Slovakia',
    'SVN': 'Slovenia', 'SWE': 'Sweden', 'SWZ': 'Eswatini',
    'SYR': 'Syria', 'TAN': 'Tanzania', 'THA': 'Thailand',
    'TJK': 'Tajikistan', 'TKM': 'Turkmenistan', 'TLS': 'East Timor',
    'TOG': 'Togo', 'TPE': 'Taiwan', 'TRI': 'Trinidad and Tobago',
    'TUN': 'Tunisia', 'TUR': 'Turkey', 'UAE': 'United Arab Emirates',
    'UGA': 'Uganda', 'UKR': 'Ukraine', 'URU': 'Uruguay',
    'USA': 'United States', 'UZB': 'Uzbekistan', 'VEN': 'Venezuela',
    'VIE': 'Vietnam', 'VIN': 'Saint Vincent and the Grenadines',
    'VIR': 'US Virgin Islands', 'WAL': 'Wales', 'YEM': 'Yemen',
    'ZAM': 'Zambia', 'ZIM': 'Zimbabwe',
    # Additional codes found unmapped
    'MWI': 'Malawi', 'REU': 'Reunion', 'DMA': 'Dominica', 'BOE': 'Bonaire',
    'MYA': 'Myanmar', 'KGZ': 'Kyrgyzstan', 'NEP': 'Nepal', 'SIM': 'Singapore',
    'IND': 'India', 'TCA': 'Turks and Caicos Islands', 'SIN': 'Singapore',
    'NCA': 'Nicaragua', 'FIJ': 'Fiji', 'LES': 'Lesotho', 'NAM': 'Namibia',
    'DJI': 'Djibouti', 'VGB': 'British Virgin Islands', 'LIE': 'Liechtenstein',
    'VAN': 'Vanuatu', 'SUD': 'Sudan', 'PNG': 'Papua New Guinea',
    'MAC': 'Macao', 'SEY': 'Seychelles', 'MDV': 'Maldives', 'LAO': 'Laos',
    'SOL': 'Solomon Islands', 'MON': 'Mongolia', 'CAM': 'Cambodia',
}


def classify_position(pos_str):
    """Map FM position string to GK/DF/MF/FW."""
    if not isinstance(pos_str, str):
        return 'MF'
    pos = pos_str.upper()
    if 'GK' in pos:
        return 'GK'
    if 'ST' in pos and ('AM' not in pos and 'M/' not in pos and 'DM' not in pos):
        return 'FW'
    if 'ST' in pos and 'AM' in pos:
        return 'FW'
    if 'AM' in pos:
        return 'FW'  # attacking mid grouped with forwards
    if pos.startswith('D') and 'M' not in pos:
        return 'DF'
    if 'WB' in pos:
        return 'DF'
    return 'MF'


def compute_fm_overall(row):
    """
    Compute a composite 'overall' rating from FM's 1-20 attributes.

    Position-aware weighting:
    - GK: heavily weight GK attributes (Ref, Han, Kic, 1v1, Cmd, Com)
    - Defenders: weight Tck, Mar, Hea, Pos, Str
    - Midfielders: weight Pas, Vis, Tec, Dec, Wor, Sta
    - Forwards: weight Fin, Dri, Pac, OtB, Cmp, Fla

    Returns value on 1-20 scale.
    """
    pos = row.get('pos_group', 'MF')

    if pos == 'GK':
        # GK-specific
        attrs = {
            'Ref': 3.0, 'Han': 2.5, '1v1': 2.5, 'Cmd': 2.0, 'Com': 1.5,
            'Kic': 1.5, 'Pun': 1.0, 'TRO': 0.5, 'Pos': 1.0, 'Agi': 1.0,
            'Dec': 1.0, 'Cnt': 1.0, 'Ant': 0.5,
        }
    elif pos == 'DF':
        attrs = {
            'Tck': 2.5, 'Mar': 2.5, 'Hea': 2.0, 'Pos': 2.0, 'Str': 1.5,
            'Jum': 1.0, 'Pac': 1.0, 'Ant': 1.5, 'Cnt': 1.5, 'Dec': 1.5,
            'Bra': 1.0, 'Cmp': 1.0, 'Pas': 0.5, 'Fir': 0.5, 'Wor': 1.0,
            'Tea': 0.5, 'Agg': 0.5, 'Sta': 0.5,
        }
    elif pos == 'FW':
        attrs = {
            'Fin': 3.0, 'Dri': 2.0, 'OtB': 2.0, 'Pac': 1.5, 'Acc': 1.5,
            'Cmp': 1.5, 'Fir': 1.5, 'Tec': 1.0, 'Fla': 0.5, 'Pas': 0.5,
            'Vis': 0.5, 'Ant': 1.0, 'Dec': 1.0, 'Bal': 0.5, 'Str': 0.5,
            'Agi': 0.5, 'Hea': 0.5,
        }
    else:  # MF
        attrs = {
            'Pas': 2.5, 'Vis': 2.0, 'Tec': 2.0, 'Dec': 2.0, 'Fir': 1.5,
            'Wor': 1.5, 'Sta': 1.5, 'Tea': 1.0, 'OtB': 1.0, 'Ant': 1.0,
            'Cmp': 1.0, 'Cnt': 1.0, 'Dri': 0.5, 'Lon': 0.5, 'Tck': 0.5,
            'Pac': 0.5, 'Bal': 0.5, 'Pos': 0.5,
        }

    total_weight = sum(attrs.values())
    weighted_sum = sum(row.get(a, 10) * w for a, w in attrs.items())
    return weighted_sum / total_weight


def calibrate_to_ea_scale(fm_overalls, ea_team_features):
    """
    Calibrate FM 1-20 overall to EA 1-99 scale using overlapping teams.

    For teams that exist in both EA and FM, we compare the top-26 averages
    and fit a linear transform: ea_overall = a * fm_overall + b
    """
    # EA 2022 features (closest year to FM23)
    ea22 = ea_team_features[ea_team_features['year'] == 2022].set_index('team')

    pairs = []
    for team, fm_avg in fm_overalls.items():
        if team in ea22.index:
            ea_avg = ea22.loc[team, 'squad_avg_overall']
            if not pd.isna(ea_avg):
                pairs.append((fm_avg, ea_avg))

    if len(pairs) < 10:
        print(f"  WARNING: only {len(pairs)} overlapping teams for calibration")
        # Fallback: rough linear transform
        return 3.5, 25.0

    fm_vals = np.array([p[0] for p in pairs])
    ea_vals = np.array([p[1] for p in pairs])

    # Linear regression: ea = a * fm + b
    A = np.vstack([fm_vals, np.ones(len(fm_vals))]).T
    a, b = np.linalg.lstsq(A, ea_vals, rcond=None)[0]

    r = np.corrcoef(fm_vals, ea_vals)[0, 1]
    print(f"  Calibration: EA = {a:.2f} * FM + {b:.2f}  (r={r:.3f}, n={len(pairs)})")

    return a, b


def parse_transfer_value(v):
    """Parse FM transfer value string to EUR."""
    if not isinstance(v, str):
        return 0
    v = v.replace('$', '').replace(',', '').replace(' ', '').strip()
    if v == '0' or v == '':
        return 0
    try:
        if 'M' in v:
            return float(v.replace('M', '')) * 1_000_000
        elif 'K' in v:
            return float(v.replace('K', '')) * 1_000
        return float(v)
    except ValueError:
        return 0


def main():
    print("Building team features from FM23 data")
    print("=" * 60)

    # Load FM23
    fm = pd.read_csv(EA_DIR / "fm23" / "merged_players (1).csv", low_memory=False)
    print(f"FM23: {len(fm)} players, {fm['Nat'].nunique()} nationalities")

    # Load existing EA features
    ea_features = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")
    ea_teams_2022 = set(ea_features[ea_features['year'] == 2022]['team'].unique())

    # Map FM codes to team names
    fm['team'] = fm['Nat'].map(FM_CODE_MAP)
    unmapped = fm['team'].isna().sum()
    if unmapped > 0:
        unmapped_codes = fm[fm['team'].isna()]['Nat'].unique()
        print(f"  Unmapped FM codes ({unmapped} players): {unmapped_codes}")
    fm = fm[fm['team'].notna()].copy()

    # Classify positions
    fm['pos_group'] = fm['Position'].apply(classify_position)
    print(f"  Position distribution: {fm['pos_group'].value_counts().to_dict()}")

    # Compute FM overall (1-20 scale)
    fm['fm_overall'] = fm.apply(compute_fm_overall, axis=1)

    # Take top-26 per nationality (same as EA pipeline)
    squads = []
    for team, group in fm.groupby('team'):
        squad = group.nlargest(min(26, len(group)), 'fm_overall').copy()
        squads.append(squad)
    fm_squads = pd.concat(squads, ignore_index=True)
    print(f"  Squads: {fm_squads['team'].nunique()} teams, {len(fm_squads)} players")

    # Compute team-level FM averages for calibration
    fm_team_avgs = fm_squads.groupby('team')['fm_overall'].mean().to_dict()

    # Calibrate FM -> EA scale
    print("\nCalibrating FM -> EA scale...")
    a, b = calibrate_to_ea_scale(fm_team_avgs, ea_features)

    # Apply calibration
    fm_squads['overall'] = (fm_squads['fm_overall'] * a + b).clip(45, 95)

    # Also create EA-like face stats by calibrating FM attributes
    # FM pace = (Acc + Pac) / 2, etc.
    fm_squads['pace'] = ((fm_squads['Acc'] + fm_squads['Pac']) / 2 * a + b).clip(40, 99)
    fm_squads['shooting'] = ((fm_squads['Fin'] + fm_squads['Lon'] + fm_squads.get('Pen', 10)) / 3 * a + b).clip(40, 99)
    fm_squads['passing'] = ((fm_squads['Pas'] + fm_squads['Vis'] + fm_squads['Fir']) / 3 * a + b).clip(40, 99)
    fm_squads['dribbling'] = ((fm_squads['Dri'] + fm_squads['Tec'] + fm_squads['Fla'] + fm_squads['Bal']) / 4 * a + b).clip(40, 99)
    fm_squads['defending'] = ((fm_squads['Tck'] + fm_squads['Mar'] + fm_squads['Pos']) / 3 * a + b).clip(40, 99)
    fm_squads['physic'] = ((fm_squads['Str'] + fm_squads['Sta'] + fm_squads['Agg']) / 3 * a + b).clip(40, 99)

    # Potential = overall + small gap (FM doesn't expose PA directly)
    fm_squads['potential'] = (fm_squads['overall'] + 2).clip(45, 99)

    # Transfer value
    fm_squads['value_eur'] = fm_squads['Transfer Value'].apply(parse_transfer_value)

    # Age and caps
    fm_squads['age'] = fm_squads['Age']
    fm_squads['caps'] = fm_squads['Caps']

    # Rename position column for compute_team_features
    fm_squads['position'] = fm_squads['pos_group']

    # Compute team features using shared pipeline
    print("\nComputing team features...")
    features = compute_team_features(fm_squads)
    features['year'] = 2022  # FM23 = closest to 2022 season
    features['source'] = 'fm23_top26'

    print(f"  Generated features for {len(features)} teams")

    # Identify teams that are NEW (not in EA data for 2022)
    new_teams = features[~features['team'].isin(ea_teams_2022)]
    existing_teams = features[features['team'].isin(ea_teams_2022)]

    print(f"  New teams (not in EA): {len(new_teams)}")
    print(f"  Overlapping teams (for validation): {len(existing_teams)}")

    # Validate: compare overlapping teams
    print("\nValidation — FM vs EA for overlapping teams (2022):")
    ea22 = ea_features[ea_features['year'] == 2022].set_index('team')
    diffs = []
    for _, row in existing_teams.iterrows():
        team = row['team']
        if team in ea22.index:
            ea_ovr = ea22.loc[team, 'squad_avg_overall']
            fm_ovr = row['squad_avg_overall']
            diff = fm_ovr - ea_ovr
            diffs.append(diff)
            if abs(diff) > 5:
                print(f"  {team:25s}  EA={ea_ovr:.1f}  FM={fm_ovr:.1f}  diff={diff:+.1f} !!!")

    diffs = np.array(diffs)
    print(f"\n  Mean diff: {diffs.mean():+.2f}")
    print(f"  Std diff:  {diffs.std():.2f}")
    print(f"  Max diff:  {max(abs(diffs)):.2f}")

    # Add ONLY new teams to existing features (don't replace EA data)
    print(f"\nAdding {len(new_teams)} new teams to team_features_by_year.csv...")

    # Add FM data for ALL years 2014-2025 as proxy (constant ratings)
    # FM23 is the only data we have for these teams — better than NaN
    all_new = []
    for proxy_year in range(2014, 2026):
        yearly = new_teams.copy()
        yearly['year'] = proxy_year
        yearly['source'] = f'fm23_proxy_{proxy_year}'
        all_new.append(yearly)

    new_rows = pd.concat(all_new, ignore_index=True)

    # Remove any existing rows for these teams/years to avoid duplicates
    for _, row in new_rows.iterrows():
        mask = (ea_features['team'] == row['team']) & (ea_features['year'] == row['year'])
        ea_features = ea_features[~mask]

    updated = pd.concat([ea_features, new_rows], ignore_index=True)
    updated = updated.sort_values(['year', 'team']).reset_index(drop=True)

    output_path = PROCESSED_DIR / "team_features_by_year.csv"
    updated.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\nSaved: {output_path}")
    print(f"Shape: {updated.shape}")
    print(f"Teams per year:")
    print(updated.groupby('year')['team'].count().to_string())

    # Show new teams ranked
    print(f"\n--- New teams from FM23 (by squad_avg_overall) ---")
    for _, r in new_teams.sort_values('squad_avg_overall', ascending=False).iterrows():
        print(f"  {r['team']:30s} | avg: {r['squad_avg_overall']:5.1f} | top3: {r['squad_top3_avg']:5.1f} | value: {r['squad_total_value']:>12,.0f}")


if __name__ == "__main__":
    main()
