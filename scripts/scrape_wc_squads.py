"""
Scrape actual World Cup squad data for 2014, 2018, 2022 from Wikipedia.
Uses the same parsing logic as scrape_squads.py but from tournament squad pages.

Output: data/processed/wc_squads_{year}.csv for each year
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import json
import time
import urllib.request
import urllib.parse
from datetime import date
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

API_BASE = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "FIFA2026Predictor/1.0 (research project; yashmori0901@gmail.com)"}

# WC pages and team name normalization
WC_PAGES = {
    2014: "2014_FIFA_World_Cup_squads",
    2018: "2018_FIFA_World_Cup_squads",
    2022: "2022_FIFA_World_Cup_squads",
}

# Wikipedia section names -> our standard team names
WIKI_TEAM_MAP = {
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "United States": "United States",
    "Czech Republic": "Czech Republic",
    "DR Congo": "DR Congo",
    "IR Iran": "Iran",
    "Costa Rica": "Costa Rica",
}


def api_get(params: dict) -> dict:
    params["format"] = "json"
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_wiki_link(text: str) -> str:
    sort_match = re.search(r'\{\{sortname\|([^|]+)\|([^|}]+)', text)
    if sort_match:
        return f"{sort_match.group(1).strip()} {sort_match.group(2).strip()}"
    match = re.search(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', text)
    return match.group(1).strip() if match else text.strip()


def parse_template_params(template_str: str) -> dict:
    params = {}
    inner = template_str.strip()
    if inner.startswith("{{"):
        inner = inner[2:]
    if inner.endswith("}}"):
        inner = inner[:-2]

    parts = []
    depth_curly = 0
    depth_square = 0
    current = ""
    for char in inner:
        if char == "{":
            depth_curly += 1
            current += char
        elif char == "}":
            depth_curly -= 1
            current += char
        elif char == "[":
            depth_square += 1
            current += char
        elif char == "]":
            depth_square -= 1
            current += char
        elif char == "|" and depth_curly == 0 and depth_square == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)

    for part in parts[1:]:
        if "=" in part:
            key, val = part.split("=", 1)
            params[key.strip().lower()] = val.strip()

    return params


def parse_dob(age_str: str) -> tuple:
    numbers = re.findall(r'\b(\d{4})\|(\d{1,2})\|(\d{1,2})', age_str)
    if numbers:
        year, month, day = int(numbers[0][0]), int(numbers[0][1]), int(numbers[0][2])
        dob = date(year, month, day)
        return dob.isoformat(), year
    return "", None


def extract_number(text: str) -> str:
    link_match = re.search(r'\[\[[^|\]]*\|(\d+)\]\]', text)
    if link_match:
        return link_match.group(1)
    num_match = re.search(r'(\d+)', text)
    if num_match:
        return num_match.group(1)
    return "0"


def parse_players_from_wikitext(wikitext: str) -> list:
    players = []
    for line in wikitext.split("\n"):
        line = line.strip()
        # Match both formats:
        # {{nat fs g player|...}} (2022+)
        # {{National football squad player|...}} (2014, 2018)
        if not re.match(r'\{\{([Nn]at fs (?:g )?player|[Nn]ational football squad player)', line):
            continue

        params = parse_template_params(line)
        name = parse_wiki_link(params.get("name", ""))
        if not name:
            continue

        dob_str, birth_year = parse_dob(params.get("age", ""))

        club_raw = params.get("club", "")
        club = parse_wiki_link(club_raw) if "[[" in club_raw else club_raw

        players.append({
            "name": name,
            "position": params.get("pos", ""),
            "dob": dob_str,
            "caps": extract_number(params.get("caps", "0")),
            "goals": extract_number(params.get("goals", "0")),
            "club": club,
            "club_country": params.get("clubnat", ""),
            "jersey_number": params.get("no", ""),
        })

    return players


def scrape_wc_year(year: int) -> list:
    page = WC_PAGES[year]
    print(f"\n{'='*60}")
    print(f"Scraping {year} FIFA World Cup squads from: {page}")

    # Get all sections
    data = api_get({"action": "parse", "page": page, "prop": "sections"})
    sections = data["parse"]["sections"]

    # Team sections are level 3
    team_sections = [(s["line"], int(s["index"])) for s in sections if s["level"] == "3" and s["line"] != "Age" and "Players" not in s["line"] and "player" not in s["line"].lower() and "representation" not in s["line"].lower() and "Coaches" not in s["line"] and "Captains" not in s["line"] and "Outfield" not in s["line"] and "Goalkeepers" not in s["line"] and "squads" not in s["line"].lower()]

    all_players = []
    for team_wiki_name, section_idx in team_sections:
        # Normalize team name
        team_name = WIKI_TEAM_MAP.get(team_wiki_name, team_wiki_name)

        print(f"  {team_name:30s}...", end=" ", flush=True)

        try:
            wdata = api_get({"action": "parse", "page": page, "prop": "wikitext", "section": section_idx})
            wikitext = wdata["parse"]["wikitext"]["*"]
            players = parse_players_from_wikitext(wikitext)

            for p in players:
                p["team"] = team_name

            all_players.extend(players)
            print(f"{len(players)} players")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.3)

    return all_players


def main():
    for year in [2014, 2018, 2022]:
        players = scrape_wc_year(year)

        output_path = DATA_DIR / f"wc_squads_{year}.csv"
        columns = ["team", "name", "position", "dob", "caps", "goals", "club", "club_country", "jersey_number"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(players)

        # Summary
        teams = set(p["team"] for p in players)
        print(f"\n  Total: {len(players)} players from {len(teams)} teams")
        print(f"  Saved: {output_path}")

        # Position breakdown
        positions = {}
        for p in players:
            pos = p["position"]
            positions[pos] = positions.get(pos, 0) + 1
        print(f"  Positions: {dict(sorted(positions.items()))}")


if __name__ == "__main__":
    main()
