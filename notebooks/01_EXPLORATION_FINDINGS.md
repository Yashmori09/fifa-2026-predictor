# 01 — Data Exploration: Findings & Decisions

This document captures what we checked, why we checked it, what we found, and what we decided for data cleaning — cell by cell.

---

## Cell 1: Imports & Setup

**Why:** Load standard data science libraries. We use `seaborn` with `darkgrid` theme for consistent chart styling. `Path` makes file references cleaner and cross-platform.

---

## Cell 2: Load All Datasets

**Why:** Load everything upfront so we can cross-reference between datasets freely. We parse dates on load (`parse_dates=["date"]`) so we don't have to convert later — avoids string comparison bugs in date logic.

**What we loaded:**
- D1: 4 files (results, goalscorers, shootouts, former_names)
- D2: 2 files (all_matches, countries_names)
- D3: 5 key files (matches, teams, tournaments, group_standings, squads)

---

## Cell 3: D1 Shape & Structure

**What we checked:** Basic dimensions, date range, column names, unique counts.

**Results:**
- **49,215 matches** from 1872-11-30 to 2026-03-31
- 9 columns: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`
- 325 unique home teams, 318 unique away teams (slight difference means some teams only ever played home or away — likely very old/defunct teams)
- 193 unique tournament types

**Key observation:** D1 has a `city` column that D2 lacks. This gives us venue-level granularity if we ever need it.

---

## Cell 4: D1 Missing Values & Dtypes

**What we checked:** Any nulls or type issues that need fixing before we can compute features.

**Results:**
- **Zero missing values across all columns** — D1 is remarkably clean
- All dtypes are correct: dates as datetime64, scores as int64, text as object, neutral as bool
- No type conversion needed

**Decision:** No imputation or null-handling required for D1. This is unusually clean for a 150-year dataset.

---

## Cell 5: Matches Per Year (Chart 1)

**What we checked:** Distribution of matches over time — are there gaps? When does coverage become dense enough for training?

**Chart description:**
![Matches per year](chart_cell7_0.png)

The bar chart shows a clear exponential growth pattern:
- **1872–1900:** Near zero — barely any international football existed. Single-digit matches per year.
- **1900–1930:** Slow ramp-up, 10–50 matches/year. Football becoming organized internationally.
- **1930–1950:** Slight growth to ~100/year, with a visible **dip during 1940–1945** (World War II — almost no international matches played).
- **1950–1990:** Steady growth from ~100 to ~500/year. Post-war boom, decolonization creating new national teams, more tournaments.
- **1990–2020:** Rapid growth to 800–1200/year. Modern era — qualifiers, continental cups, Nations League, friendlies all exploding.
- **2020:** Sharp dip (COVID-19 pandemic — international football largely suspended).
- **2021–2025:** Recovery to 1000+/year.
- **2026:** Only ~160 so far (data only through March 2026).

**Decision:** 
- Pre-1990 data is too sparse for reliable feature engineering (form, H2H) but still valuable for ELO warmup.
- The ELO system needs the full timeline to converge — we'll feed it everything from 1872.
- For actual model training features (form, H2H windows), we'll use 1990+ data where match density is sufficient.
- COVID dip in 2020 might distort "recent form" calculations — we should be aware of this edge case.

---

## Cell 6: Top 15 Tournament Types (Chart 2)

**What we checked:** Which tournament types dominate the dataset — this directly impacts our tournament category mapping for ELO K-values.

**Chart description:**
Horizontal bar chart showing match counts per tournament type. Dominant categories:
1. **Friendly** — by far the most common (~17,000+ matches), roughly a third of all data
2. **FIFA World Cup qualification** — second most (~7,000+)
3. **UEFA Euro qualification** — third (~3,000+)
4. **African Cup of Nations qualification** — significant volume
5. **FIFA World Cup** — the actual tournament (fewer matches since it's only every 4 years, 32 teams until now)
6. **African Cup of Nations**, **AFC Asian Cup qualification**, **Gold Cup**, **CECAFA Cup**, **CFU Caribbean Cup**, etc.

**Key observations:**
- Friendlies are 35%+ of data. They're low-stakes and less predictive — we'll assign them the lowest ELO K-value (20).
- Qualification matches are a huge chunk. These are competitive and should get K=40.
- Actual World Cup matches are relatively rare (~900 in the dataset). Continental finals get K=50, WC finals get K=60.
- There are many small regional tournaments (CECAFA Cup, CFU Caribbean Cup, Merdeka Tournament, etc.) that we need to categorize as "other" (K=30).

**Decision:** We need to map all 193 tournament names into 5 categories:
1. `world_cup` (K=60) — "FIFA World Cup"
2. `continental_final` (K=50) — "Copa América", "UEFA Euro", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "Confederations Cup"
3. `qualifier` (K=40) — anything with "qualification" or "qualifier" in the name
4. `other_competitive` (K=30) — regional cups, Nations League, Olympics, etc.
5. `friendly` (K=20) — "Friendly"

---

## Cell 7: Score Distributions (Chart 3)

**What we checked:** How goals are distributed — this tells us about class balance (for outcome prediction) and what a reasonable scoreline model should produce.

**Chart description:** Three histograms side by side:

**Home Score Distribution (blue):**
- Right-skewed, peaking at 1 goal (~14,000 matches)
- 0 goals is second most common (~12,000)
- Rapid dropoff after 2 goals
- Long tail extending to 10+ (rare blowouts)

**Away Score Distribution (orange):**
- More aggressively right-skewed, peaking at **0 goals** (~18,000 matches)
- Away teams score 0 in a huge proportion of matches — confirms home advantage
- Drops off faster than home scores — away teams rarely score 3+

**Total Goals Distribution (purple):**
- Peaks at 2 total goals (~11,000 matches)
- 1 and 3 goals also very common (~9,000 each)
- Mean: 2.94 goals per match
- Roughly Poisson-like distribution — validates using Poisson regression for scoreline prediction

**Outcome percentages:**
- Home win: **49.0%**
- Draw: **22.8%**
- Away win: **28.3%**

**Key observations:**
- Strong home advantage — home teams win nearly half the time, away teams only 28%.
- Draw is the minority class at 23% — this will be the hardest to predict (class imbalance).
- The score distributions look Poisson — good, this means a Poisson model for scoreline prediction is theoretically justified.
- Average home score (1.76) is significantly higher than away score (1.18) — a 0.58 goal home advantage.

**Decision:**
- We may need to handle class imbalance for the draw class (consider class weighting or SMOTE).
- Poisson regression is a valid choice for the scoreline prediction model.
- Home/away distinction and neutral ground flag are clearly important features — the data confirms this.

---

## Cell 8: Neutral Ground Breakdown

**What we checked:** How many matches are on neutral ground — relevant because neutral matches remove home advantage.

**Results:**
- **Neutral: 12,976 (26.4%)**
- **Home/Away: 36,239 (73.6%)**

**Key observation:** A quarter of all matches are on neutral ground. This is higher than expected — likely because World Cup and continental tournament matches are often at neutral venues. This makes the `neutral` flag an important feature — on neutral ground, we shouldn't apply home advantage adjustments.

**Decision:** Include `neutral` as a binary feature. Also, in ELO computation, we only add the +100 home advantage adjustment when `neutral=False`.

---

## Cell 9: D2 Shape & Structure

**What we checked:** Same basic exploration for D2 to understand differences from D1.

**Results:**
- **51,485 matches** — 2,270 more than D1
- Same date range: 1872–2026
- 8 columns (same as D1 minus `city`)
- 278 home teams, 286 away teams — **fewer teams than D1** (D1 has 325/318)
- 226 unique tournament types — **more tournament types than D1** (193)
- Zero missing values

**Key observations:**
- D2 has more matches but fewer teams. This means D2 excludes some microstates/non-FIFA teams that D1 includes (CONIFA teams, regional teams like Catalonia, Basque Country, etc.).
- D2 has more tournament type granularity (226 vs 193) — likely splits some tournaments D1 groups together (e.g., "World Cup qualifier" vs "FIFA World Cup qualification").

---

## Cell 10: D1 vs D2 Overlap Analysis

**What we checked:** How much these two independently-maintained datasets overlap — critical for the merge strategy.

**Results:**
- **D1 total unique matches: 49,214**
- **D2 total unique matches: 51,407**
- **Overlapping: 37,058** (the core they both agree on)
- **Only in D1: 12,156**
- **Only in D2: 14,349**
- **Combined unique: 63,563**

**Key observation:** Only ~72% overlap. That's significant — each dataset has 12-14k matches the other doesn't. Merging both gives us **63,563 matches** vs. ~50k from either alone. That's a 27% increase in training data.

**Decision:** Definitely merge both datasets. Use (date, home_team, away_team) as the match key for deduplication. When both have the same match, prefer D1's row (has `city` column).

---

## Cell 11: Team Name Differences

**What we checked:** Which teams exist in one dataset but not the other — reveals naming mismatches and coverage differences.

**Results:**
- D1 has **105 teams** not in D2 — mostly CONIFA/non-FIFA teams: Abkhazia, Catalonia, Basque Country, Corsica, Isle of Man, Padania, etc.
- Also includes naming differences: D1 has `China PR`, `Czech Republic` where D2 uses `China`, `Czechia`

**Key observations:**
- Most D1-only teams are non-FIFA entities that won't appear in World Cup prediction. Safe to keep them for ELO but they won't affect our 2026 predictions.
- The naming differences (`China PR` vs `China`, `Czech Republic` vs `Czechia`) are a merge-blocking issue — if we don't fix these, a team gets split into two separate entities.

**Decision:** Build a canonical name mapping that resolves all variants to one name. We'll use D1's naming convention as the base (since D1 is our primary dataset) and map D2 names to match.

---

## Cell 12: D2-Only Matches

**What we checked:** What kind of matches does D2 have that D1 doesn't — to understand what we gain by merging.

**Results:**
- **14,410 D2-only matches**
- Top tournament types: Friendly (4,197), World Cup qualifier (1,156), Friendly tournament (870), Olympic Games qualifier (735), World Cup (444)
- Date range: 1882 to 2026

**Key observation:** D2 has 444 World Cup matches that D1 doesn't! This is significant — possibly women's World Cup or matches with different team name variants. Also 1,156 additional World Cup qualifiers. The friendlies are less critical, but qualifiers and WC matches are valuable training data.

**Decision:** The D2-only matches are not just noise — they include competitive matches we genuinely want. Confirms the merge is worth doing. The tournament name differences ("World Cup qualifier" in D2 vs "FIFA World Cup qualification" in D1) need to be harmonized during categorization.

---

## Cell 13-14: D3 World Cup Database

**What we checked:** What the Fjelstul WC database gives us beyond flat match results.

**Results:**
- **30 tournaments** (1930–2022) — includes both Men's and Women's World Cups
- **1,248 matches** with rich metadata
- **85 teams** with confederation mappings
- **13,843 players** in squads data
- 37 columns per match — including stage info, extra time, penalties, stadium, etc.

**Stage breakdown:** 880 group stage, 113 R16, 70 QF, 38 SF, 29 finals — maps perfectly to WC tournament structure.

**Confederation data:** UEFA (39 teams), CAF (13), AFC (12), CONCACAF (11), CONMEBOL (9), OFC (1) — only 85 teams total (those that have appeared in a WC). We'll need to supplement this for the full team list.

**Key observation:** D3 includes Women's World Cups (1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019). We need to filter to Men's only for our predictions.

**Decision:** 
- Use D3's `teams.csv` as the starting point for confederation mapping, but supplement it — 85 teams isn't enough, we need confederations for all ~300+ teams.
- Filter D3 to Men's World Cup only.
- D3's stage/knockout metadata will be useful for tournament simulation logic.

---

## Cell 15: Team Name Mapping Files

**What we checked:** The built-in name mapping files from D1 and D2.

**D1 `former_names.csv`:** 36 mappings of historical → current names (Dahomey→Benin, Soviet Union→Russia, Zaïre→DR Congo, etc.). These handle name changes over time — a team that played as "Soviet Union" in 1988 is the same entity as "Russia" in 1994.

**D2 `countries_names.csv`:** 47 mappings where `original_name != current_name`. Broader coverage than D1's file. Also includes color codes (useful for UI later).

**Decision:** Combine both mapping files to build the most complete name normalization dictionary. D2's file has more mappings, D1's file has date ranges (useful if we ever want to preserve historical names for display).

---

## Cell 16: Cross-Dataset Name Check

**What we checked:** Specific known-problematic team names across all three datasets.

**Results:**
| Team | D1 | D2 | D3 |
|---|---|---|---|
| South Korea | Yes | Yes | Yes |
| Korea Republic | No | No | No |
| China PR | Yes | No | No |
| China | No | Yes | Yes |
| Czech Republic | Yes | No | Yes |
| Czechia | No | Yes | No |
| Ivory Coast | Yes | Yes | Yes |
| United States | Yes | Yes | Yes |

**D3 teams not in D1:** China, Dutch East Indies, East Germany, Serbia and Montenegro, Soviet Union, West Germany, Zaire — all historical names that D1 has already modernized.

**Key naming conflicts to resolve:**
1. `China PR` (D1) vs `China` (D2/D3) → standardize to `China PR` (FIFA's official name)
2. `Czech Republic` (D1/D3) vs `Czechia` (D2) → standardize to `Czech Republic`
3. D3 historical names → map to modern equivalents (West Germany→Germany, Soviet Union→Russia, etc.)

---

## Summary: What We Decided for Data Cleaning

Based on all findings above, here's the cleaning plan for `02_data_cleaning.ipynb`:

### 1. Team Name Standardization
- Build a master name mapping dictionary combining D1's `former_names.csv` and D2's `countries_names.csv`
- Add manual fixes: `China` → `China PR`, `Czechia` → `Czech Republic`
- Apply to both D1 and D2 before merging
- Map D3 historical names (West Germany, Soviet Union, etc.) to modern equivalents

### 2. Merge D1 + D2
- Standardize team names in both datasets first
- Merge on `(date, home_team, away_team, home_score, away_score)`
- When duplicates exist, keep D1's row (has `city` column)
- When a match is only in D2, take D2's row (add empty `city` column)
- Expected result: ~63,000+ unique matches

### 3. Tournament Categorization
- Map all 193+ tournament names into 5 categories for ELO K-values:
  - `world_cup` (K=60)
  - `continental_final` (K=50)
  - `qualifier` (K=40)
  - `other_competitive` (K=30)
  - `friendly` (K=20)
- Harmonize D1/D2 tournament name differences (e.g., "FIFA World Cup qualification" vs "World Cup qualifier")

### 4. Confederation Mapping
- Start with D3's 85-team confederation data
- Supplement with web lookup or manual mapping for remaining teams
- This gives us confederation as a feature for every team

### 5. D3 Filtering
- Filter to Men's World Cup only (exclude 1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019 Women's tournaments)
- Keep the rich metadata (stages, penalties, extra time) for tournament simulation logic

### 6. Output
- `processed/matches_clean.csv` — single merged, deduplicated, standardized dataset
- `processed/team_confederations.csv` — team → confederation mapping
- `processed/tournament_categories.csv` — tournament name → category mapping

### 7. Edge Cases to Watch
- COVID-19 gap in 2020 — may distort form calculations
- Very old matches (pre-1900) — useful for ELO warmup only, not for training features
- Non-FIFA teams (CONIFA, regional) — keep in data but won't appear in 2026 predictions
- Draw class imbalance (23%) — may need handling during model training
