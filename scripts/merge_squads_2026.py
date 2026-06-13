"""
Merge real 2026 WC squad list (from Wikipedia) with player ratings:
  1) Fresh fcratings.com EA FC 26 data (scrape_fcratings.py output) — priority
  2) Existing frontend/src/data/squad_players.json (fallback for players
     not in fcratings but already in old data)
  3) No data — include with empty rating fields

Output:
  frontend/src/data/squad_players.json (overwritten)
  data/processed/merge_report.csv      (per-player audit)
"""

import csv
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WC_PATH = ROOT / "data/processed/wc_squads_2026.csv"
FCR_PATH = ROOT / "data/processed/fcratings.csv"
OLD_JSON = ROOT / "frontend/src/data/squad_players.json"
NEW_JSON = ROOT / "frontend/src/data/squad_players.json"
REPORT = ROOT / "data/processed/merge_report.csv"
BACKUP = ROOT / "frontend/src/data/squad_players.backup.json"

TODAY = date(2026, 6, 11)  # WC opening day


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", "", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def age_from_dob(dob_iso):
    if not dob_iso:
        return None
    try:
        y, m, d = [int(x) for x in dob_iso.split("-")]
        dob = date(y, m, d)
    except Exception:
        return None
    age = TODAY.year - dob.year - ((TODAY.month, TODAY.day) < (dob.month, dob.day))
    return age


def rep_from_ovr(ovr):
    if not ovr:
        return None
    if ovr >= 88:
        return 5
    if ovr >= 84:
        return 4
    if ovr >= 79:
        return 3
    if ovr >= 73:
        return 2
    return 1


def val_from_ovr(ovr, age):
    if not ovr:
        return None
    base = {90: 90_000_000, 85: 50_000_000, 80: 22_000_000,
            75: 8_000_000, 70: 2_500_000, 65: 800_000, 60: 200_000}
    val = 500_000
    for thresh in sorted(base.keys(), reverse=True):
        if ovr >= thresh:
            val = base[thresh]
            break
    # Age discount: peak around 24-28, decline after 30
    if age:
        if age >= 33:
            val = int(val * 0.4)
        elif age >= 30:
            val = int(val * 0.6)
        elif age >= 28:
            val = int(val * 0.8)
        elif age <= 20:
            val = int(val * 1.2)
    return val


def int_or_none(v):
    if v in (None, "", "None"):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def load_fcratings():
    """Return {(team, norm_name): fcratings_row}"""
    if not FCR_PATH.exists():
        return {}
    out = {}
    with open(FCR_PATH) as f:
        for row in csv.DictReader(f):
            out[(row["team"], norm(row["name"]))] = row
    return out


def load_old_json():
    """Return {(team, norm_name): old_player_dict}"""
    if not OLD_JSON.exists():
        return {}
    with open(OLD_JSON) as f:
        data = json.load(f)
    out = {}
    for team, players in data.items():
        for p in players:
            out[(team, norm(p["name"]))] = p
    return out


def build_player(wc_row, fcr_row, old_row):
    """Compose final player dict using fcratings > old_json fallback."""
    name = wc_row["name"]
    pos = wc_row["position"] or (old_row or {}).get("pos") or ""
    club = wc_row["club"] or (old_row or {}).get("club") or ""
    age = age_from_dob(wc_row.get("dob"))
    if age is None and old_row:
        age = old_row.get("age")
    caps = int_or_none(wc_row.get("caps")) or (old_row or {}).get("caps") or 0
    goals = int_or_none(wc_row.get("goals")) or (old_row or {}).get("goals") or 0

    p = {
        "name": name,
        "pos": pos,
        "age": age,
        "caps": caps,
        "goals": goals,
        "club": club,
    }

    # Source of ratings: fcratings > old_json > None
    source = "none"
    if fcr_row and fcr_row.get("ovr"):
        source = "fcratings"
        ovr = int_or_none(fcr_row["ovr"])
        p["ovr"] = ovr
        p["pot"] = old_row.get("pot") if old_row else ovr  # potential not in fcratings; use old or fallback to ovr
        p["rep"] = (old_row.get("rep") if old_row else None) or rep_from_ovr(ovr)
        p["val"] = (old_row.get("val") if old_row else None) or val_from_ovr(ovr, age)
        if pos == "GK":
            # GK detailed stats from fcratings (0-99 scale)
            for k_fcr, k_out in [("div", "div"), ("han", "han"), ("kic", "kic"),
                                 ("gkp", "gkp"), ("ref", "ref")]:
                v = int_or_none(fcr_row.get(k_fcr))
                if v is not None:
                    p[k_out] = v
                elif old_row and old_row.get(k_out) is not None:
                    p[k_out] = old_row[k_out]
        else:
            for k in ["pac", "sho", "pas", "dri", "defe", "phy"]:
                v = int_or_none(fcr_row.get(k))
                if v is not None:
                    p[k] = v
                elif old_row and old_row.get(k) is not None:
                    p[k] = old_row[k]
    elif old_row:
        source = "old_json"
        for k in ["ovr", "pot", "rep", "val", "pac", "sho", "pas", "dri", "defe", "phy",
                  "div", "han", "kic", "gkp", "ref"]:
            if k in old_row and old_row[k] is not None:
                p[k] = old_row[k]
    # else: leave rating fields absent

    return p, source


def main():
    print("Loading inputs...")
    with open(WC_PATH) as f:
        wc = list(csv.DictReader(f))
    fcr = load_fcratings()
    old = load_old_json()
    print(f"  Real 2026 squad players: {len(wc)}")
    print(f"  fcratings entries: {len(fcr)}")
    print(f"  Old JSON entries: {len(old)}")

    # Backup old JSON
    if OLD_JSON.exists() and not BACKUP.exists():
        BACKUP.write_text(OLD_JSON.read_text())
        print(f"  Backed up old JSON to {BACKUP}")

    new_data = {}
    report_rows = []
    counts = {"fcratings": 0, "old_json": 0, "none": 0}

    for row in wc:
        team = row["team"]
        n = norm(row["name"])
        fcr_row = fcr.get((team, n))
        old_row = old.get((team, n))
        player, source = build_player(row, fcr_row, old_row)
        new_data.setdefault(team, []).append(player)
        counts[source] += 1
        report_rows.append({
            "team": team,
            "name": row["name"],
            "position": row["position"],
            "club": row["club"],
            "source": source,
            "ovr": player.get("ovr", ""),
        })

    # Sort each team's roster by ovr desc (matches existing JSON convention)
    for team in new_data:
        new_data[team].sort(key=lambda p: -(p.get("ovr") or 0))

    # Write outputs
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, separators=(",", ":"))
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "name", "position", "club", "source", "ovr"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"\nWrote {NEW_JSON} ({len(new_data)} teams, {sum(len(v) for v in new_data.values())} players)")
    print(f"Wrote {REPORT}")
    print(f"\nSource breakdown:")
    for k, v in counts.items():
        print(f"  {k:<12} {v}")


if __name__ == "__main__":
    main()
