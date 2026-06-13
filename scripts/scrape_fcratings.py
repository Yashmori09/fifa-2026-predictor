"""
Scrape FC 26 player ratings from fcratings.com for 2026 WC squad members
not already in frontend/src/data/squad_players.json.

No Cloudflare. Uses fcratings.com's static search-index JSON to resolve
names → URLs, then fetches each player page over plain HTTPS.

Resumable: writes data/processed/fcratings.csv incrementally.

Usage:
  python scripts/scrape_fcratings.py [--limit N] [--threads K]
"""

import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MISSING_PATH = ROOT / "data/processed/missing_players_2026.csv"
INDEX_PATH = ROOT / "data/processed/fcratings_index.json"
OUT_PATH = ROOT / "data/processed/fcratings.csv"
FAILED_PATH = ROOT / "data/processed/fcratings_failed.csv"

INDEX_URL = "https://www.fcratings.com/wp-content/plugins/fc-instant-search/assets/search-index.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

OUT_COLS = [
    "team", "name", "position", "club", "fcr_url", "matched_name", "matched_club",
    "ovr", "pos_url",
    "pac", "sho", "pas", "dri", "defe", "phy",
    "gk_overall", "div", "han", "kic", "gkp", "ref",
]


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", "", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_or_fetch_index():
    if INDEX_PATH.exists() and INDEX_PATH.stat().st_size > 1_000_000:
        with open(INDEX_PATH) as f:
            return json.load(f)
    print(f"Fetching search index from {INDEX_URL} ...")
    data = json.loads(fetch(INDEX_URL))
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(data, f)
    print(f"  saved {len(data)} entries to {INDEX_PATH}")
    return data


def build_player_lookup(index):
    """Build normalized-name -> [{name, club, url, ovr}] map for men's players only."""
    lookup = {}
    for e in index:
        if e.get("t") != "player" or e.get("g", 0) != 0:
            continue
        n = norm(e.get("n", ""))
        if not n:
            continue
        lookup.setdefault(n, []).append({
            "name": e["n"], "club": e.get("s", ""), "url": e["u"], "ovr": e.get("r", 0)
        })
    return lookup


def find_match(player, lookup):
    """Find best fcratings entry for a player from our missing list."""
    name = player["name"]
    club_n = norm(player.get("club", ""))
    name_n = norm(name)

    # Exact normalized match
    candidates = lookup.get(name_n, [])

    # If exact match — return best (preferring club match if multiple)
    if candidates:
        return _pick(candidates, club_n)

    # Fallback 1: substring match — fcratings name fully contained in ours or vice versa
    # (handles "Mohamed Salah" vs "Mohamed Salah Ghaly" etc.)
    sub_candidates = []
    for k, v in lookup.items():
        if k != name_n and (name_n in k or k in name_n):
            # Require shared first AND last word to avoid junk matches
            our_words = name_n.split()
            their_words = k.split()
            if len(our_words) >= 2 and len(their_words) >= 2:
                if our_words[0] == their_words[0] and our_words[-1] == their_words[-1]:
                    sub_candidates.extend(v)
    if sub_candidates:
        return _pick(sub_candidates, club_n)

    # Fallback 2: last-name match ONLY if club matches
    # (avoids Luis Romo -> Rafael Romo type errors)
    if club_n:
        parts = name_n.split()
        if len(parts) >= 2:
            last = parts[-1]
            club_matches = []
            for k, v in lookup.items():
                if k.endswith(" " + last) or k == last:
                    for c in v:
                        if norm(c["club"]) and (club_n in norm(c["club"]) or norm(c["club"]) in club_n):
                            club_matches.append(c)
            if len(club_matches) == 1:
                return club_matches[0]

    return None


def _pick(candidates, club_n):
    """Pick best candidate, preferring exact club match."""
    if len(candidates) == 1:
        return candidates[0]
    if club_n:
        for c in candidates:
            cn = norm(c["club"])
            if cn and (club_n == cn or club_n in cn or cn in club_n):
                return c
        # Loose: any word overlap
        club_words = set(club_n.split())
        for c in candidates:
            c_words = set(norm(c["club"]).split())
            if club_words & c_words:
                return c
    return candidates[0]


def parse_player_page(html):
    """Extract stats from an fcratings player page."""
    out = {}

    # Overall from description meta
    m = re.search(r"is rated (\d+) overall in FC 26", html)
    if m:
        out["ovr"] = int(m.group(1))

    # Outfield 6 attrs + optional GK column from chart data + labels
    labels_m = re.search(r"labels:\s*\[([^\]]+)\]", html)
    data_m = re.search(r"data:\s*\[([\d,\s]+)\]", html)
    if labels_m and data_m:
        labels = [s.strip(' "\'') for s in labels_m.group(1).split(",")]
        values = [int(v.strip()) for v in data_m.group(1).split(",") if v.strip()]
        if len(labels) == len(values):
            label_map = {"PAC": "pac", "SHO": "sho", "PAS": "pas", "DRI": "dri",
                         "DEF": "defe", "PHY": "phy", "GK": "gk_overall"}
            for lab, val in zip(labels, values):
                key = label_map.get(lab)
                if key:
                    out[key] = val

    # GK detailed stats — pattern: <span...>GK Diving</span>...<span...>NN</span>
    for label, key in [("Diving", "div"), ("Handling", "han"), ("Kicking", "kic"),
                       ("Positioning", "gkp"), ("Reflexes", "ref")]:
        m = re.search(
            rf">GK\s+{label}</span>\s*<span[^>]*>(\d+)</span>",
            html,
        )
        if m:
            out[key] = int(m.group(1))

    # Position url
    m = re.search(r"/positions/([a-z-]+)", html)
    if m:
        out["pos_url"] = m.group(1)

    return out


def process_player(player, lookup):
    pick = find_match(player, lookup)
    if not pick:
        return {"status": "no_match", "player": player}

    try:
        html = fetch(pick["url"])
        stats = parse_player_page(html)
    except Exception as e:
        return {"status": "fetch_error", "player": player, "pick": pick, "error": str(e)}

    if not stats.get("ovr"):
        return {"status": "no_ovr", "player": player, "pick": pick}

    row = {
        "team": player["team"],
        "name": player["name"],
        "position": player.get("position", ""),
        "club": player.get("club", ""),
        "fcr_url": pick["url"],
        "matched_name": pick["name"],
        "matched_club": pick["club"],
        **stats,
    }
    return {"status": "ok", "row": row}


def main():
    limit = None
    threads = 8
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--limit":
            limit = int(args.pop(0))
        elif a == "--threads":
            threads = int(args.pop(0))

    index = load_or_fetch_index()
    lookup = build_player_lookup(index)
    print(f"Built lookup: {len(lookup)} unique normalized names")

    with open(MISSING_PATH) as f:
        missing = list(csv.DictReader(f))

    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            done = {(r["team"], norm(r["name"])) for r in csv.DictReader(f)}

    failed = set()
    if FAILED_PATH.exists():
        with open(FAILED_PATH) as f:
            failed = {(r["team"], norm(r["name"])) for r in csv.DictReader(f)}

    todo = [p for p in missing if (p["team"], norm(p["name"])) not in done and (p["team"], norm(p["name"])) not in failed]
    if limit:
        todo = todo[:limit]
    print(f"Total: {len(missing)} | done: {len(done)} | failed: {len(failed)} | todo: {len(todo)}")
    if not todo:
        return

    out_f = open(OUT_PATH, "a", newline="", encoding="utf-8")
    fail_f = open(FAILED_PATH, "a", newline="", encoding="utf-8")
    out_w = csv.DictWriter(out_f, fieldnames=OUT_COLS)
    fail_w = csv.DictWriter(fail_f, fieldnames=["team", "name", "status", "extra"])
    if not OUT_PATH.stat().st_size:
        out_w.writeheader()
    if not FAILED_PATH.stat().st_size:
        fail_w.writeheader()

    ok = miss = err = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(process_player, p, lookup): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            p = futs[fut]
            if res["status"] == "ok":
                row = res["row"]
                out_w.writerow({k: row.get(k, "") for k in OUT_COLS})
                out_f.flush()
                ok += 1
                tag = "OK"
                detail = f"ovr={row.get('ovr','?')} matched={res['row']['matched_name']}"
            else:
                fail_w.writerow({"team": p["team"], "name": p["name"], "status": res["status"], "extra": str(res.get("pick", res.get("error", "")))[:80]})
                fail_f.flush()
                miss += 1 if res["status"] == "no_match" else 0
                err += 1 if res["status"] != "no_match" else 0
                tag = res["status"].upper()
                detail = str(res.get("pick", res.get("error", "")))[:60]
            if i % 20 == 0 or i == len(todo):
                elapsed = time.time() - start
                print(f"  [{i}/{len(todo)}] ok={ok} miss={miss} err={err}  elapsed={elapsed:.0f}s")
            else:
                print(f"  [{i}/{len(todo)}] {tag:<10} {p['team']:<15} {p['name']:<28} {detail}")

    out_f.close()
    fail_f.close()
    print(f"\nDone. OK: {ok} | no_match: {miss} | other failures: {err}")


if __name__ == "__main__":
    main()
