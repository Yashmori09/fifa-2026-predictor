"""
Extend squad_ratings_2026.csv with FM23 data for players not in FC 26.

For each player with NaN `overall` in squad_ratings_2026.csv:
  1. Find them in FM23 by name + nationality.
  2. Compute composite FM "overall" from position-weighted 1-20 attributes.
  3. Calibrate to EA scale using players matched in both FC 26 and FM 23.
  4. Map FM attributes to EA's 6 (pac/sho/pas/dri/defe/phy) + GK attrs.

Writes back to squad_ratings_2026.csv (in-place).
"""

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQUAD_PATH = ROOT / "data/processed/squad_ratings_2026.csv"
FM_PATH = ROOT / "data/ea_fc/fm23/merged_players (1).csv"

# Reuse compute_fm_overall + nationality map from build_fm_features
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fm_features import compute_fm_overall, classify_position as get_pos_group, FM_CODE_MAP

# FM nationality (3-letter code) → our team name reverse map
NAT_TO_TEAM = dict(FM_CODE_MAP)  # already FM code -> our team name


def strip_diacritics(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s):
    if not isinstance(s, str):
        return ""
    s = strip_diacritics(s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s)


# EA 6 attributes → FM attribute composites
# (weights for each FM attr that contributes)
OUTFIELD_ATTR_MAP = {
    "pace":      {"Pac": 0.6, "Acc": 0.4},
    "shooting":  {"Fin": 0.5, "Lon": 0.25, "Tec": 0.15, "Pen": 0.1},
    "passing":   {"Pas": 0.5, "Vis": 0.25, "Tec": 0.25},
    "dribbling": {"Dri": 0.5, "Tec": 0.25, "Bal": 0.15, "Agi": 0.1},
    "defending": {"Mar": 0.4, "Tck": 0.4, "Pos": 0.1, "Hea": 0.1},
    "physic":    {"Str": 0.4, "Sta": 0.3, "Bra": 0.15, "Jum": 0.15},
}

# EA GK attributes → FM
GK_ATTR_MAP = {
    "div": {"Aer": 0.6, "Agi": 0.4},
    "han": {"Han": 1.0},
    "kic": {"Kic": 0.7, "Thr": 0.3},
    "gkp": {"Cmd": 0.5, "Pos": 0.5},
    "ref": {"Ref": 1.0},
}


def fm_composite(row, mapping):
    total = sum(mapping.values())
    s = 0.0
    for attr, w in mapping.items():
        v = row.get(attr)
        if pd.isna(v):
            v = 10
        s += float(v) * w
    return s / total


def main():
    squads = pd.read_csv(SQUAD_PATH, low_memory=False)
    fm = pd.read_csv(FM_PATH, low_memory=False)
    print(f"Squads: {len(squads)}  FM23: {len(fm)}")

    # Add pos_group to FM
    fm["pos_group"] = fm["Position"].apply(get_pos_group)

    # Normalize FM names + nationality (FM uses 3-letter codes in Nat column)
    fm["_name_norm"] = fm["Name"].apply(norm)
    fm["_team"] = fm["Nat"].map(NAT_TO_TEAM)

    # Identify unmatched
    needs_fm = squads[squads["overall"].isna()].copy()
    print(f"Players needing FM23 fallback: {len(needs_fm)}")

    # 1) Compute FM composite "overall" (1-20 scale) for all FM players
    fm["fm_overall_1_20"] = fm.apply(compute_fm_overall, axis=1)

    # 2) Calibrate FM 1-20 → EA 1-99 using overlapping players
    # Find FM players that also exist in squads with EA overall (matched in FC 26)
    matched = squads[squads["overall"].notna()].copy()
    matched["_name_norm"] = matched["name"].apply(norm)

    # Match by (team, name) — but FM is global, squads is just 2026 WC players
    fm_lookup = fm.set_index(["_team", "_name_norm"])["fm_overall_1_20"]
    pairs = []
    for _, row in matched.iterrows():
        key = (row["team"], norm(row["name"]))
        if key in fm_lookup.index:
            fm_val = fm_lookup.loc[key]
            if isinstance(fm_val, pd.Series):
                fm_val = fm_val.iloc[0]
            pairs.append((float(fm_val), float(row["overall"])))

    print(f"Overlap players for calibration: {len(pairs)}")
    if len(pairs) < 30:
        a, b = 3.5, 25.0  # rough fallback
        print(f"  Too few pairs — using fallback EA = {a}*FM + {b}")
    else:
        fm_vals = np.array([p[0] for p in pairs])
        ea_vals = np.array([p[1] for p in pairs])
        A = np.vstack([fm_vals, np.ones(len(fm_vals))]).T
        a, b = np.linalg.lstsq(A, ea_vals, rcond=None)[0]
        r = np.corrcoef(fm_vals, ea_vals)[0, 1]
        print(f"  Calibration: EA_ovr = {a:.3f}*FM + {b:.3f}  (r={r:.3f})")

    # Per-attr calibration for the 6 EA face stats
    attr_cal = {}
    for ea_attr, fm_map in OUTFIELD_ATTR_MAP.items():
        # Compute FM composite for this attr for every FM player
        fm[f"_fm_{ea_attr}"] = fm.apply(lambda r: fm_composite(r, fm_map), axis=1)
        fm_attr_lookup = fm.set_index(["_team", "_name_norm"])[f"_fm_{ea_attr}"]
        pairs_a = []
        for _, row in matched.iterrows():
            key = (row["team"], norm(row["name"]))
            if key in fm_attr_lookup.index:
                fmv = fm_attr_lookup.loc[key]
                if isinstance(fmv, pd.Series): fmv = fmv.iloc[0]
                eav = row.get(ea_attr)
                if pd.notna(eav):
                    pairs_a.append((float(fmv), float(eav)))
        if len(pairs_a) >= 30:
            fmv = np.array([p[0] for p in pairs_a])
            eav = np.array([p[1] for p in pairs_a])
            A_ = np.vstack([fmv, np.ones(len(fmv))]).T
            aa, bb = np.linalg.lstsq(A_, eav, rcond=None)[0]
            attr_cal[ea_attr] = (aa, bb, len(pairs_a))
            print(f"  {ea_attr}: EA = {aa:.2f}*FM + {bb:.2f}  (n={len(pairs_a)})")
        else:
            attr_cal[ea_attr] = (a, b, 0)  # fall back to overall calibration

    # GK attr calibration (very few matched outfield→GK overlap; use same)
    gk_cal = {}
    for ea_attr, fm_map in GK_ATTR_MAP.items():
        fm[f"_fm_{ea_attr}"] = fm.apply(lambda r: fm_composite(r, fm_map), axis=1)
        # EA stores GK attrs as goalkeeping_diving etc.
        col = {"div":"goalkeeping_diving", "han":"goalkeeping_handling",
               "kic":"goalkeeping_kicking", "gkp":"goalkeeping_positioning",
               "ref":"goalkeeping_reflexes"}.get(ea_attr)
        fm_attr_lookup = fm.set_index(["_team", "_name_norm"])[f"_fm_{ea_attr}"]
        pairs_a = []
        for _, row in matched.iterrows():
            if row.get("position") != "GK":
                continue
            key = (row["team"], norm(row["name"]))
            if key in fm_attr_lookup.index:
                fmv = fm_attr_lookup.loc[key]
                if isinstance(fmv, pd.Series): fmv = fmv.iloc[0]
                eav = row.get(col)
                if pd.notna(eav):
                    pairs_a.append((float(fmv), float(eav)))
        if len(pairs_a) >= 10:
            fmv = np.array([p[0] for p in pairs_a])
            eav = np.array([p[1] for p in pairs_a])
            A_ = np.vstack([fmv, np.ones(len(fmv))]).T
            aa, bb = np.linalg.lstsq(A_, eav, rcond=None)[0]
            gk_cal[ea_attr] = (aa, bb)
            print(f"  GK {ea_attr}: EA = {aa:.2f}*FM + {bb:.2f}  (n={len(pairs_a)})")
        else:
            gk_cal[ea_attr] = (a, b)

    # 3) Now fill in unmatched players from FM23
    filled = 0
    not_found = []
    for idx, row in needs_fm.iterrows():
        team = row["team"]
        name = row["name"]
        n = norm(name)
        # Try matching in FM by (team, name)
        cands = fm[(fm["_team"] == team) & (fm["_name_norm"] == n)]
        if len(cands) == 0:
            # Try without team constraint
            cands = fm[fm["_name_norm"] == n]
        if len(cands) == 0:
            # Fuzzy: last-name match within team
            parts = n.split()
            if len(parts) >= 2:
                last = parts[-1]
                cands = fm[(fm["_team"] == team) & (fm["_name_norm"].str.endswith(" " + last))]
        if len(cands) == 0:
            not_found.append((team, name))
            continue

        # Pick best by FM overall
        fm_row = cands.sort_values("fm_overall_1_20", ascending=False).iloc[0]
        fm_ovr = fm_row["fm_overall_1_20"]
        ea_ovr = max(40, min(95, round(a * fm_ovr + b)))
        squads.at[idx, "overall"] = ea_ovr
        squads.at[idx, "potential"] = ea_ovr  # no potential info from this comp.

        if row.get("position") == "GK":
            for ea_attr in ["div","han","kic","gkp","ref"]:
                col = {"div":"goalkeeping_diving","han":"goalkeeping_handling",
                       "kic":"goalkeeping_kicking","gkp":"goalkeeping_positioning",
                       "ref":"goalkeeping_reflexes"}[ea_attr]
                fmv = fm_row[f"_fm_{ea_attr}"]
                aa, bb = gk_cal[ea_attr]
                val = max(20, min(95, round(aa * fmv + bb)))
                squads.at[idx, col] = val
        else:
            for ea_attr in ["pace","shooting","passing","dribbling","defending","physic"]:
                fmv = fm_row[f"_fm_{ea_attr}"]
                aa, bb, _ = attr_cal[ea_attr]
                val = max(20, min(95, round(aa * fmv + bb)))
                squads.at[idx, ea_attr] = val
        filled += 1

    print(f"\nFilled {filled} previously-unmatched players from FM23")
    print(f"Still not found: {len(not_found)}")
    for t, n in not_found[:20]:
        print(f"  {t}: {n}")

    squads.to_csv(SQUAD_PATH, index=False)
    print(f"\nWrote {SQUAD_PATH}")
    final_matched = squads["overall"].notna().sum()
    print(f"Final coverage: {final_matched}/{len(squads)} = {final_matched/len(squads)*100:.1f}%")


if __name__ == "__main__":
    main()
