"""
B5 — Scrape FBref Big-5 European Leagues stats via local Camoufox bypass server.

Pages:
  - keepersadv (PSxG, PSxG-GA per keeper)
  - defense    (tackles, interceptions, aerial duels per player)
  - shooting   (xG breakdown including PK / FK contribution)

Seasons: 2022-2023, 2023-2024, 2024-2025

Output:
  data/raw/fbref/{stat}_{season}.html      ← cached raw HTML
  data/raw/fbref/{stat}_{season}.csv       ← parsed table

Run the bypass server first:
  cd /tmp/cfbypass && DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib .venv/bin/python server.py
"""
import time
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/fbref"
RAW.mkdir(parents=True, exist_ok=True)

BYPASS = "http://localhost:8000/html"
SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
STATS = ["keepersadv", "defense", "shooting"]

# FBref Big-5 player-aggregate URL template
URL = "https://fbref.com/en/comps/Big5/{season}/{stat}/players/{season}-Big-5-European-Leagues-Stats"


def fetch(url: str, html_path: Path, retries: int = 2) -> str:
    """GET through the bypass server, with on-disk caching."""
    if html_path.exists() and html_path.stat().st_size > 50_000:
        print(f"  cached: {html_path.name} ({html_path.stat().st_size:,} bytes)")
        return html_path.read_text()
    for attempt in range(retries + 1):
        try:
            r = requests.get(BYPASS, params={"url": url}, timeout=180)
            r.raise_for_status()
            if "Just a moment" in r.text:
                print(f"  attempt {attempt+1}: Cloudflare not bypassed, retrying...")
                time.sleep(5)
                continue
            html_path.write_text(r.text)
            print(f"  fetched: {html_path.name} ({len(r.text):,} bytes)")
            return r.text
        except Exception as e:
            print(f"  attempt {attempt+1} error: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch {url}")


def parse_main_table(html: str) -> pd.DataFrame:
    """FBref hides large tables inside HTML comments. Strip comments, then
    find the largest table that looks like a player table."""
    html_clean = html.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html_clean))
    best = None
    for t in tables:
        flat = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
        if "Player" in flat and len(t) >= 50:
            if best is None or len(t) > len(best):
                best = t
    if best is None:
        raise RuntimeError("No player table found")
    # Flatten multi-index columns
    if isinstance(best.columns, pd.MultiIndex):
        # Disambiguate duplicates by prefixing with parent group
        new_cols = []
        seen = {}
        for parent, child in best.columns:
            base = child if not parent.startswith("Unnamed") else child
            label = base
            if label in seen:
                label = f"{parent}_{base}"
            seen[label] = True
            new_cols.append(label)
        best.columns = new_cols
    # Drop repeated header rows
    best = best[best["Player"] != "Player"].reset_index(drop=True)
    return best


def main():
    for season in SEASONS:
        for stat in STATS:
            print(f"\n=== {stat} {season} ===")
            url = URL.format(season=season, stat=stat)
            html_path = RAW / f"{stat}_{season}.html"
            csv_path = RAW / f"{stat}_{season}.csv"

            html = fetch(url, html_path)
            df = parse_main_table(html)
            df.to_csv(csv_path, index=False)
            print(f"  parsed {len(df)} rows × {len(df.columns)} cols → {csv_path.name}")
            time.sleep(3)  # be polite between requests

    # Summary
    print("\n=== Summary ===")
    for f in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):>4} rows × {len(df.columns):>2} cols")


if __name__ == "__main__":
    main()
