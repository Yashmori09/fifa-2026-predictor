"""Check FM23 coverage for missing teams and estimate impact on training data."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

df = pd.read_csv(DATA_DIR / "ea_fc" / "fm23" / "merged_players (1).csv", low_memory=False)
print(f"FM23: {len(df)} players, {df['Nat'].nunique()} nationalities")

FM_CODE_MAP = {
    'AFG': 'Afghanistan', 'ALB': 'Albania', 'ALG': 'Algeria', 'AND': 'Andorra',
    'ANG': 'Angola', 'ARG': 'Argentina', 'ARM': 'Armenia', 'ARU': 'Aruba',
    'ATG': 'Antigua and Barbuda', 'AUS': 'Australia', 'AUT': 'Austria',
    'AZE': 'Azerbaijan', 'BAH': 'Bahamas', 'BAN': 'Bangladesh', 'BDI': 'Burundi',
    'BEL': 'Belgium', 'BEN': 'Benin', 'BER': 'Bermuda', 'BFA': 'Burkina Faso',
    'BHR': 'Bahrain', 'BIH': 'Bosnia and Herzegovina', 'BLR': 'Belarus',
    'BLZ': 'Belize', 'BOL': 'Bolivia', 'BOT': 'Botswana',
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
}

missing_teams = [
    'Malawi','Madagascar','Thailand','Uganda','Botswana','El Salvador','Lesotho',
    'Mozambique','Eswatini','Oman','Namibia','Bahrain','Gibraltar','Vietnam',
    'Tanzania','Seychelles','Philippines','Malaysia','Maldives','Yemen','Palestine',
    'Cambodia','Myanmar','Faroe Islands','Kazakhstan','Andorra','Liechtenstein',
    'Mauritius','Sudan','Malta','San Marino','North Korea','Burundi','Bangladesh',
    'Dominica','Grenada','Mauritania','Ethiopia','Barbados','Laos','East Timor',
    'Kuwait','Lebanon','Libya','Singapore','Nepal','South Sudan','Rwanda','Taiwan',
    'Nicaragua','Hong Kong','Antigua and Barbuda','Guyana','Afghanistan','Cuba',
    'Kyrgyzstan','Martinique','Guatemala','Saint Lucia','Mongolia','Tajikistan','Niger'
]

team_to_fm = {v: k for k, v in FM_CODE_MAP.items()}

found = []
not_found = []

for team in sorted(missing_teams):
    fm_code = team_to_fm.get(team)
    if fm_code and fm_code in df['Nat'].values:
        count = len(df[df['Nat'] == fm_code])
        found.append((team, fm_code, count))
    else:
        not_found.append(team)

print(f"\nFOUND IN FM23: {len(found)}/{len(missing_teams)}")
print(f"\n{'Team':30s} | {'FM Code':>7s} | {'Players':>7s}")
print("-" * 50)
for team, code, count in sorted(found, key=lambda x: -x[2]):
    marker = " (strong)" if count >= 20 else " (ok)" if count >= 10 else " (thin)"
    print(f"{team:30s} | {code:>7s} | {count:>7d}{marker}")

print(f"\nNOT FOUND: {len(not_found)}")
for t in not_found:
    print(f"  {t}")

# Coverage impact
train = pd.read_csv(PROCESSED_DIR / "train_dc.csv")
train['date'] = pd.to_datetime(train['date'])
m2014 = train[train['date'].dt.year >= 2014]

tf = pd.read_csv(PROCESSED_DIR / "team_features_by_year.csv")
ea_teams = set(tf['team'].unique())
fm_teams = set(t for t, _, _ in found)
all_covered = ea_teams | fm_teams

both_ea = sum(1 for _, r in m2014.iterrows()
              if r['home_team'] in ea_teams and r['away_team'] in ea_teams)
both_all = sum(1 for _, r in m2014.iterrows()
               if r['home_team'] in all_covered and r['away_team'] in all_covered)

print(f"\n{'='*50}")
print(f"COVERAGE IMPACT (2014+ matches)")
print(f"{'='*50}")
print(f"Total 2014+ matches:    {len(m2014)}")
print(f"Before (EA only):       {both_ea} ({both_ea/len(m2014)*100:.1f}%)")
print(f"After (EA + FM):        {both_all} ({both_all/len(m2014)*100:.1f}%)")
print(f"Gain:                   +{both_all - both_ea} matches (+{(both_all-both_ea)/len(m2014)*100:.1f}%)")
print(f"\nTeams covered: EA={len(ea_teams)}, FM new={len(fm_teams)}, Total={len(all_covered)}")
