"""
Download StatsBomb Open Data for the international tournaments most relevant
to WC 2026 predictions.

Tournaments pulled (competition_id, season_id):
  43, 106  → World Cup 2022
  43,   3  → World Cup 2018
  55, 282  → Euro 2024
  55,  43  → Euro 2020
  223, 282 → Copa America 2024
  1267, 107 → AFCON 2023

Saves:
  data/raw/statsbomb/matches/{competition_id}_{season_id}.json
  data/raw/statsbomb/events/{match_id}.json

Uses GitHub raw URLs — no scraping, no auth needed.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/statsbomb"
MATCHES_DIR = RAW / "matches"
EVENTS_DIR = RAW / "events"
MATCHES_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

TOURNAMENTS = [
    (43, 106, "WC 2022"),
    (43, 3, "WC 2018"),
    (55, 282, "Euro 2024"),
    (55, 43, "Euro 2020"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]


def fetch_json(url: str, dest: Path, retries: int = 3) -> dict:
    if dest.exists() and dest.stat().st_size > 100:
        return json.loads(dest.read_text())
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_text(r.text)
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download_tournament(comp_id: int, season_id: int, name: str):
    print(f"\n=== {name} (cid={comp_id} sid={season_id}) ===")
    matches_url = f"{BASE}/matches/{comp_id}/{season_id}.json"
    matches_path = MATCHES_DIR / f"{comp_id}_{season_id}.json"
    matches = fetch_json(matches_url, matches_path)
    print(f"  matches: {len(matches)}")

    match_ids = [m["match_id"] for m in matches]

    # Download events in parallel — but stay polite with a small worker pool
    done = 0
    failed = []

    def grab(mid):
        url = f"{BASE}/events/{mid}.json"
        dest = EVENTS_DIR / f"{mid}.json"
        try:
            fetch_json(url, dest)
            return mid, None
        except Exception as e:
            return mid, str(e)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(grab, mid) for mid in match_ids]
        for f in as_completed(futures):
            mid, err = f.result()
            done += 1
            if err:
                failed.append((mid, err))
            if done % 20 == 0:
                print(f"  {done}/{len(match_ids)} events fetched")

    print(f"  ✓ {done - len(failed)}/{len(match_ids)} events ok")
    if failed:
        print(f"  ✗ {len(failed)} failed: {failed[:3]}")


def main():
    for comp_id, season_id, name in TOURNAMENTS:
        download_tournament(comp_id, season_id, name)

    # Summary
    total_matches = len(list(MATCHES_DIR.glob("*.json")))
    total_events = len(list(EVENTS_DIR.glob("*.json")))
    total_size = sum(p.stat().st_size for p in EVENTS_DIR.glob("*.json")) / 1024 / 1024
    print(f"\n=== Total: {total_matches} match files, {total_events} event files, {total_size:.0f} MB ===")


if __name__ == "__main__":
    main()
