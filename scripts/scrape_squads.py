"""
Scrape current squad data for all 48 FIFA 2026 World Cup teams from Wikipedia.
Uses MediaWiki API to fetch structured wikitext, then parses player templates.

Output: data/processed/squads_2026.csv
Fields: team, name, position, dob, age, caps, goals, club, club_country, jersey_number
"""

import re
import csv
import json
import time
import urllib.request
import urllib.parse
from datetime import date, datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Wikipedia page names for each of the 48 WC teams
TEAM_PAGES = {
    "Mexico": "Mexico_national_football_team",
    "South Africa": "South_Africa_national_soccer_team",
    "South Korea": "South_Korea_national_football_team",
    "Czech Republic": "Czech_Republic_national_football_team",
    "Canada": "Canada_men's_national_soccer_team",
    "Bosnia and Herzegovina": "Bosnia_and_Herzegovina_national_football_team",
    "Qatar": "Qatar_national_football_team",
    "Switzerland": "Switzerland_national_football_team",
    "Brazil": "Brazil_national_football_team",
    "Morocco": "Morocco_national_football_team",
    "Haiti": "Haiti_national_football_team",
    "Scotland": "Scotland_national_football_team",
    "United States": "United_States_men's_national_soccer_team",
    "Paraguay": "Paraguay_national_football_team",
    "Australia": "Australia_men's_national_soccer_team",
    "Turkey": "Turkey_national_football_team",
    "Germany": "Germany_national_football_team",
    "Curaçao": "Curaçao_national_football_team",
    "Ivory Coast": "Ivory_Coast_national_football_team",
    "Ecuador": "Ecuador_national_football_team",
    "Netherlands": "Netherlands_national_football_team",
    "Japan": "Japan_national_football_team",
    "Sweden": "Sweden_men's_national_football_team",
    "Tunisia": "Tunisia_national_football_team",
    "Belgium": "Belgium_national_football_team",
    "Egypt": "Egypt_national_football_team",
    "Iran": "Iran_national_football_team",
    "New Zealand": "New_Zealand_men's_national_football_team",
    "Spain": "Spain_national_football_team",
    "Cape Verde": "Cape_Verde_national_football_team",
    "Saudi Arabia": "Saudi_Arabia_national_football_team",
    "Uruguay": "Uruguay_national_football_team",
    "France": "France_national_football_team",
    "Senegal": "Senegal_national_football_team",
    "Iraq": "Iraq_national_football_team",
    "Norway": "Norway_national_football_team",
    "Argentina": "Argentina_national_football_team",
    "Algeria": "Algeria_national_football_team",
    "Austria": "Austria_national_football_team",
    "Jordan": "Jordan_national_football_team",
    "Portugal": "Portugal_national_football_team",
    "DR Congo": "DR_Congo_national_football_team",
    "Uzbekistan": "Uzbekistan_national_football_team",
    "Colombia": "Colombia_national_football_team",
    "England": "England_national_football_team",
    "Croatia": "Croatia_national_football_team",
    "Ghana": "Ghana_national_football_team",
    "Panama": "Panama_national_football_team",
}

API_BASE = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "FIFA2026Predictor/1.0 (research project; yashmori0901@gmail.com)"}


def api_get(params: dict) -> dict:
    """Make a GET request to MediaWiki API."""
    params["format"] = "json"
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_squad_section(page: str) -> int | None:
    """Find the section index for 'Current squad' on a Wikipedia page."""
    data = api_get({"action": "parse", "page": page, "prop": "sections"})
    sections = data.get("parse", {}).get("sections", [])

    # Exact match first (any nesting level)
    for section in sections:
        title = section.get("line", "").strip().lower()
        if title in ("current squad", "current roster", "current squad and coaching staff"):
            return int(section["index"])

    # Broader match
    for section in sections:
        title = section.get("line", "").strip().lower()
        if "current" in title and ("squad" in title or "roster" in title or "player" in title):
            return int(section["index"])

    # Last resort: look for "Players" parent section and check subsections
    for section in sections:
        title = section.get("line", "").strip().lower()
        if title == "players":
            # Return the next section which is likely "Current squad"
            players_idx = int(section["index"])
            for sub in sections:
                if int(sub["index"]) > players_idx and int(sub.get("level", 0)) > int(section.get("level", 0)):
                    sub_title = sub.get("line", "").strip().lower()
                    if "squad" in sub_title or "roster" in sub_title:
                        return int(sub["index"])
            # If no squad subsection found, return the Players section itself
            return players_idx

    return None


def fetch_squad_wikitext(page: str, section: int) -> str:
    """Fetch wikitext for a specific section."""
    data = api_get({"action": "parse", "page": page, "prop": "wikitext", "section": section})
    return data.get("parse", {}).get("wikitext", {}).get("*", "")


def parse_wiki_link(text: str) -> str:
    """Extract display name from [[Link|Display]], [[Name]], or {{sortname|First|Last}} format."""
    # Handle {{sortname|First|Last|dab=...}} template
    sort_match = re.search(r'\{\{sortname\|([^|]+)\|([^|}]+)', text)
    if sort_match:
        return f"{sort_match.group(1).strip()} {sort_match.group(2).strip()}"
    # Handle [[Link|Display]] or [[Name]]
    match = re.search(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', text)
    return match.group(1).strip() if match else text.strip()


def parse_template_params(template_str: str) -> dict:
    """Parse key=value pairs from a wikitext template call."""
    params = {}
    # Remove outer {{ and }}
    inner = template_str.strip()
    if inner.startswith("{{"):
        inner = inner[2:]
    if inner.endswith("}}"):
        inner = inner[:-2]

    # Split by | but respect nested {{ }} and [[ ]]
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

    # First part is template name, rest are params
    for part in parts[1:]:
        if "=" in part:
            key, val = part.split("=", 1)
            params[key.strip().lower()] = val.strip()

    return params


def parse_dob(age_str: str) -> tuple[str, int | None]:
    """Parse DOB from {{birth date and age|...}} or {{bda|...}} template. Returns (dob_str, age)."""
    # Extract year, month, day from the template
    numbers = re.findall(r'\b(\d{4})\|(\d{1,2})\|(\d{1,2})', age_str)
    if numbers:
        year, month, day = int(numbers[0][0]), int(numbers[0][1]), int(numbers[0][2])
        dob = date(year, month, day)
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return dob.isoformat(), age
    return "", None


def extract_number(text: str) -> str:
    """Extract a number from text that might contain wiki links."""
    # Handle [[List of...|56]] format
    link_match = re.search(r'\[\[[^|\]]*\|(\d+)\]\]', text)
    if link_match:
        return link_match.group(1)
    # Handle plain number
    num_match = re.search(r'(\d+)', text)
    if num_match:
        return num_match.group(1)
    return "0"


def parse_players(wikitext: str) -> list[dict]:
    """Parse all player entries from squad wikitext."""
    players = []

    # Parse line by line — each player is one line like {{nat fs g player|...}}
    for line in wikitext.split("\n"):
        line = line.strip()
        if not re.match(r'\{\{[Nn]at fs (?:g )?player', line):
            continue
        template = line
        params = parse_template_params(template)

        name = parse_wiki_link(params.get("name", ""))
        if not name:
            continue

        dob_str, age = parse_dob(params.get("age", ""))

        club_raw = params.get("club", "")
        club = parse_wiki_link(club_raw) if "[[" in club_raw else club_raw

        players.append({
            "name": name,
            "position": params.get("pos", ""),
            "dob": dob_str,
            "age": age if age is not None else "",
            "caps": extract_number(params.get("caps", "0")),
            "goals": extract_number(params.get("goals", "0")),
            "club": club,
            "club_country": params.get("clubnat", ""),
            "jersey_number": params.get("no", ""),
        })

    return players


def scrape_team(team_name: str, page: str) -> list[dict]:
    """Scrape squad for a single team."""
    print(f"  Fetching {team_name}...", end=" ", flush=True)

    try:
        section_idx = find_squad_section(page)
        if section_idx is None:
            print(f"WARNING: No 'Current squad' section found!")
            return []

        wikitext = fetch_squad_wikitext(page, section_idx)
        players = parse_players(wikitext)

        # Add team name to each player
        for p in players:
            p["team"] = team_name

        print(f"{len(players)} players")
        return players

    except Exception as e:
        print(f"ERROR: {e}")
        return []


def main():
    print(f"Scraping squads for {len(TEAM_PAGES)} teams from Wikipedia...\n")

    all_players = []
    failed_teams = []

    for team_name, page in TEAM_PAGES.items():
        players = scrape_team(team_name, page)
        if players:
            all_players.extend(players)
        else:
            failed_teams.append(team_name)
        time.sleep(0.5)  # Be polite to Wikipedia

    # Save to CSV
    output_path = DATA_DIR / "squads_2026.csv"
    columns = ["team", "name", "position", "dob", "age", "caps", "goals", "club", "club_country", "jersey_number"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_players)

    print(f"\n{'='*60}")
    print(f"Total: {len(all_players)} players from {len(TEAM_PAGES) - len(failed_teams)}/{len(TEAM_PAGES)} teams")
    print(f"Saved to: {output_path}")

    if failed_teams:
        print(f"\nFailed teams ({len(failed_teams)}): {', '.join(failed_teams)}")

    # Summary stats
    positions = {}
    for p in all_players:
        pos = p["position"]
        positions[pos] = positions.get(pos, 0) + 1
    print(f"\nPosition breakdown: {dict(sorted(positions.items()))}")


if __name__ == "__main__":
    main()
