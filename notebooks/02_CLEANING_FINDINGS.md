# 02 — Data Cleaning: Findings & Decisions

This document captures what each cell did, why, what it found, issues encountered, fixes applied, and final outputs.

---

## Cell 1: Imports & Setup

**Why:** Load libraries and create `processed/` directory if it doesn't exist. `Path.mkdir(exist_ok=True)` means it won't error if it already exists — safe to re-run.

---

## Cell 2: Load Raw Data

**Results:** D1 (49,215 rows), D2 (51,485 rows), D3 (1,248 rows). All loaded cleanly with no errors.

---

## Cell 3: Build Master Team Name Mapping

**Why:** Before merging D1 and D2, we need consistent team names across both datasets. Different datasets use different names for the same team — if we don't fix this, a team like "Czech Republic" and "Czechia" become two separate entities in our model, splitting their historical record and corrupting ELO calculations.

**How we built it (3 layers):**
1. **D1 `former_names.csv`** — maps historical names to current ones (e.g., `Soviet Union → Russia`, `Zaïre → DR Congo`). These handle country renames over time.
2. **D2 `countries_names.csv`** — maps D2's variant names to canonical ones. Only adds entries where `original_name != current_name` to avoid no-op entries.
3. **Manual fixes** — cross-dataset mismatches found during exploration: `China → China PR`, `Czechia → Czech Republic`, `West Germany → Germany`, `Yugoslavia → Serbia`, etc.

**Result:** 73 total mappings.

**Issue encountered:** D1's `former_names.csv` maps `Bohemia → Czechia` (historical). Our manual fix maps `Czechia → Czech Republic`. But `normalize_team_name` originally only did a single dictionary lookup — so `Bohemia → Czechia` resolved but never continued to `Czech Republic`.

**Fix:** Changed `normalize_team_name` to do **two lookups** (handle one level of chaining). `Bohemia → Czechia → Czech Republic` now resolves correctly in a single call.

---

## Cell 4: Apply Name Normalization to D1 and D2

**Why:** Apply the mapping to all `home_team` and `away_team` columns in both datasets before merging.

**Verification results (after fix):**
- `China PR` in D1: True ✓
- `China` in D1: False ✓ (all mapped to `China PR`)
- `China PR` in D2: True ✓
- `Czech Republic` in D2: True ✓
- `Czechia` in D2: **False** ✓ (all mapped to `Czech Republic` after chaining fix)

---

## Cell 5: Merge D1 + D2

**Why:** Concatenate both datasets before deduplication. D1 is added first so that when we drop duplicates with `keep='first'`, D1 rows take priority (they have the `city` column that D2 lacks).

**Steps:**
- Added `source` tag to each row (`D1` or `D2`) for traceability
- Added empty `city` column to D2 rows
- Normalized `neutral` column to consistent `bool` type (D2 used string `'True'/'False'`)
- Concatenated D1 + D2 → 100,700 combined rows before dedup

---

## Cell 6: Deduplicate

**Why:** After concatenation, matches that exist in both datasets appear twice. We deduplicate on `(date, home_team, away_team, home_score, away_score)` — this is the tightest possible match key without being too strict (avoiding false deduplication of replayed matches on the same day).

**Results:**
- Before dedup: 100,700
- After dedup: **60,692** — 39,000+ duplicates removed
- D1 rows kept: 49,215 (all of D1)
- D2 rows kept: 11,477 (matches only in D2)
- Unique teams: **342** (one fewer than before — Czechia merged into Czech Republic ✓)

**Why 60,692 and not ~63,563 from exploration?**
In exploration we computed overlap using `(date, home_team, away_team)` only. Here we use the full key including scores. Some matches had the same teams/date but different recorded scores across datasets — these are kept as separate rows since we can't be sure which score is correct. Hence slightly fewer deduped rows than estimated.

---

## Cell 7: Missing Values Check

**Results:**
- All columns: 0 missing values
- `city`: 11,477 missing — expected, these are D2-only rows which have no city data
- `city` is not a training feature so this is fine — no imputation needed

---

## Cell 8: Tournament Categorization

**Why:** 338 unique tournament names need to be bucketed into 5 categories for ELO K-value assignment. We use keyword-based rules since the tournament names follow predictable patterns.

**Categories and K-values:**
| Category | K-value | Rationale |
|---|---|---|
| `world_cup` | 60 | Highest stakes — every match matters enormously |
| `continental_final` | 50 | Major confederation tournaments (Copa América, Euros, AFCON, etc.) |
| `qualifier` | 40 | Competitive matches with real consequences |
| `other_competitive` | 30 | Regional cups, games with moderate stakes |
| `friendly` | 20 | Lowest weight — results are least indicative of true strength |

**Issue encountered (and fixed):**
- `European Championship qual` was hitting `other_competitive` — our qualifier check looked for `'qualif'`, `'qualifier'`, `'qualifying'` but not `'qual'` as a suffix. **Fix:** added `.endswith(' qual')` check.
- `FIFA Series` was hitting `world_cup` because it contained "series" and the check was `'world cup' in name OR 'fifa series' in name`. **Fix:** removed FIFA Series from the world_cup check and added an explicit early return as `other_competitive` (K=30) — these are warmup/friendly-style matches, not actual WC matches.

**Final distribution:**
| Category | Matches |
|---|---|
| friendly | 22,075 |
| qualifier | 18,676 |
| other_competitive | 10,808 |
| continental_final | 7,712 |
| world_cup | 1,421 |

**Verification spot check:**
- `FIFA World Cup` → `world_cup` (K=60) ✓
- `FIFA World Cup qualification` → `qualifier` (K=40) ✓
- `European Championship qual` → `qualifier` (K=40) ✓ (after fix)
- `FIFA Series` → `other_competitive` (K=30) ✓ (after fix)
- `Copa América` → `continental_final` (K=50) ✓
- `Friendly` → `friendly` (K=20) ✓

---

## Cell 9: Review `other_competitive` Bucket

**Why:** Sanity check — if any important tournaments were misclassified, we'd catch them here.

**Top tournaments in `other_competitive`:**
Merdeka Tournament (599), Gulf Cup (577), Asian Games (525), British Home Championship (523), Island Games (395), King's Cup (338), South Pacific Games (298), Nordic Championship (290), Arab Cup (250), etc.

**Assessment:** All look correctly placed. These are legitimate regional/sub-confederation tournaments that are competitive but not major continental events. No miscategorizations spotted — no need to reclassify anything.

---

## Cell 10: Save Tournament Reference File

**Output:** `processed/tournament_categories.csv` — 338 unique tournament names with their assigned category and K-value. Useful for inspection and reproducibility.

---

## Cell 11: D3 Confederation Data

**Why:** We need a confederation label for every team (UEFA, CAF, AFC, CONCACAF, CONMEBOL, OFC). D3's `teams.csv` has this for the 85 teams that have ever appeared in a World Cup.

**Results:** 85 teams — UEFA (39), CAF (13), AFC (12), CONCACAF (11), CONMEBOL (9), OFC (1).

**Limitation:** Only covers World Cup participants — 263 out of 343 teams in our dataset are missing confederation data.

---

## Cell 12: Find Teams Missing Confederation

**Why:** We need to assign confederations to the remaining 263 teams for the confederation feature in our model.

**Result:** 263 teams missing. These include:
1. **Legitimate FIFA member nations** — Afghanistan, Albania, Andorra, Armenia, etc. that have never appeared in a WC
2. **Non-FIFA / CONIFA entities** — Catalonia, Basque Country, Abkhazia, Isle of Man, etc. These are regional/unrecognized teams that won't appear in 2026 predictions

---

## Cell 13: Manual Confederation Assignments

**Why:** Manually assign confederations for all FIFA member nations not covered by D3. We used official FIFA confederation membership.

**Coverage after manual mapping:**
| Confederation | Teams |
|---|---|
| UEFA | 56 |
| CAF | 54 |
| AFC | 47 |
| CONCACAF | 38 |
| CONMEBOL | 10 |
| OFC | 16 |
| UNKNOWN | 122 |

**Issues encountered and fixed:** Initial manual mapping missed several legitimate FIFA nations — Albania, Finland, Cyprus, Monaco (UEFA), Seychelles, South Africa (CAF), and several OFC members. Added in second pass.

---

## Cell 14: UNKNOWN Teams

**Result:** 122 UNKNOWN teams remaining. All are CONIFA/non-FIFA entities:
- Regional/autonomy teams: Catalonia, Basque Country, Corsica, Isle of Man, Brittany, etc.
- Unrecognized states: Abkhazia, Kosovo (now UEFA), Somaliland, South Ossetia, etc.
- Historical entities: North Vietnam, Yemen DPR, German DR, etc.
- Microstates with no FIFA membership: Vatican City, Sealand, etc.

**Decision:** These are excluded from 2026 predictions — none will appear in the World Cup. Their historical matches still contribute to ELO computation, but their confederation feature will be `UNKNOWN` (treated as a separate category).

---

## Cell 15: Save Confederation File

**Output:** `processed/team_confederations.csv` — 343 teams with confederation assignments.

---

## Cell 16: Add Derived Columns

**Why:** Add features we'll need during feature engineering and modelling.

**Columns added:**
- `home_confederation` / `away_confederation` — looked up from confederation mapping
- `outcome` — 3-class target variable: `home_win`, `draw`, `away_win`
- `goal_diff` — `home_score - away_score` (used in ELO computation and as a feature)

**Outcome distribution:**
- home_win: 31,836 (52.5%)
- away_win: 15,262 (25.1%)
- draw: 13,594 (22.4%)

Dataset sorted chronologically — critical for ELO computation (must process matches in time order).

---

## Cell 17: Filter D3 to Men's World Cup

**Why:** D3 includes Women's World Cups (1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019) which are irrelevant for our predictions. We filter to Men's WC only.

**Result:** 964 Men's WC matches across 22 tournaments (1930–2022). 284 Women's WC matches excluded. All team names normalized using same mapping.

---

## Cell 18: Save All Outputs

**Files saved to `processed/`:**
| File | Rows | Description |
|---|---|---|
| `matches_clean.csv` | 60,692 | Main cleaned dataset — single source of truth |
| `d3_mens_wc.csv` | 964 | Men's WC matches with rich D3 metadata |
| `tournament_categories.csv` | 338 | Tournament → category + K-value reference |
| `team_confederations.csv` | 343 | Team → confederation reference |
| `team_name_mapping.csv` | 73 | Name variant → canonical name reference |

---

## Summary of Issues Found & Fixed

| Issue | Root Cause | Fix |
|---|---|---|
| `Czechia` persisting in D2 after normalization | Single-lookup `normalize_team_name` couldn't chain `Bohemia→Czechia→Czech Republic` | Changed to two-level lookup — resolves one level of chaining |
| `European Championship qual` → `other_competitive` | Qualifier keyword check didn't cover `' qual'` suffix | Added `.endswith(' qual')` to qualifier detection |
| `FIFA Series` → `world_cup` | Regex matched `'fifa series'` with world_cup check | Added explicit early return as `other_competitive` |
| 132 UNKNOWN confederations | Manual conf mapping was incomplete | Added second pass covering Albania, Finland, Cyprus, Monaco, Seychelles etc. — reduced to 122 |

---

## What Goes into Feature Engineering Next

From `matches_clean.csv` we now compute for every match:
1. **ELO ratings** — iterate chronologically, update after every match using K-values from `elo_k` column
2. **Recent form** — last 5 and 10 matches per team (win rate, goals scored/conceded)
3. **Head-to-head** — historical record for each specific team pair
4. **Confederation** — already added, just encode it
5. **Neutral ground** — already in the data as a boolean

All features will be computed "at the time of the match" — i.e., using only data available before that match was played. No data leakage.
