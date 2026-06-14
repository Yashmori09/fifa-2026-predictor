# Phase 3 — Modernized model with player-level signals

> A focused retrain that addresses the structural blind spots surfaced during the 2026 WC.

## Why a Phase 3

Phase 2 ships with documented limitations:
- Training data goes back to 1884. ~10K matches pre-1990 reflect a different game (no substitutions, no professionalism, no scouting). Even 1990-2010 matches predate the data-driven era. Only the ~13K matches from 2010 onward really resemble current football.
- All squad signals are **static**. EA FC ratings + caps tell you who's in the squad, not who's in form or who's injured.
- The model can't see **chemistry**: how often the same XI has played together, how many starters share a club, how long the coach has been in charge.
- The 2026 WC has three **co-hosts** (USA / Canada / Mexico) playing on home soil. The model treats every WC match as neutral. Real bookmakers don't.
- Live performance on the first matches of the WC (e.g., USA 4-1 Paraguay — model picked Paraguay 39%) suggests these blind spots matter.

Phase 3 fixes the inputs and the data window, not the learner. XGB+RF stays.

---

## Goal

**Predict every 2026 FIFA World Cup match — outcome and scoreline — and eventually the tournament winner.**

A model that:
1. Trains on the modern era only.
2. Incorporates per-player current form (not just static ratings).
3. Recognises team chemistry through squad continuity + same-club concentration.
4. Treats host nations as playing at home.
5. Includes match context (rest days, travel, fatigue).
6. Validates against literature benchmarks (target RPS ≤ 0.20, ECE < 0.025).
7. Ships before the knockout stage if possible, otherwise becomes the post-WC retrospective model.

## Locked decisions (Phase 1)

These three were debated and locked on 2026-06-14:

| Decision | Choice | Why |
|---|---|---|
| **Success bar** | Beat "pick higher squad rating" baseline by **5%+ outcome accuracy** | The dumb non-ML rule hits ~52% on international football. Our ML earns its complexity only by beating it by a meaningful margin (target ≥60%). Bookmaker-odds parity (~55-60%) is unrealistic in this scope. |
| **Historical lineups** | Scrape FBref for actual lineups of past international matches | Approach B from the original plan. Time-consuming but the only honest way to compute trailing-12-month player form features for historical training matches. Top-26-by-caps proxy is acceptable fallback if specific matches can't be sourced. |
| **Architecture** | Two-track hybrid: XGB+RF classifier for outcome + Dixon-Coles bivariate Poisson for scoreline | The current Phase 2 approach reverse-engineers Poisson lambdas from outcome probabilities (approximation). A proper Dixon-Coles model gives real goal expectancies with the rho correction — academic standard for football scoreline prediction. We already have Dixon-Coles infrastructure from Phase 1. |

---

## Gaps to close

### Tier 1 — Player-level signals (the biggest miss)
| Signal | What it measures | Why it matters |
|---|---|---|
| Current club form | Goals, assists, minutes, xG/xA per 90 over last 1–2 seasons | Single highest-impact feature in modern football models |
| Player availability | Injured / suspended / fit | Spain without Pedri is a different team |
| Position-by-position form | GK save %, FW conversion rate, DF clean sheets | Position-specific signals beat aggregate squad ratings |

### Tier 2 — Team chemistry
| Signal | Source | Notes |
|---|---|---|
| Same-club concentration | `squad_players.json` (already have) | % of squad sharing top 1/3 clubs. Easy win. |
| Lineup continuity | Last 2 years of national team lineups | Real chemistry proxy — counts shared minutes between XI |
| Coach tenure | Wikipedia | Long-tenured coaches = baked-in tactics |
| Tactical fingerprint (optional) | Recent match stats (pass %, build-up style, defensive line height) | Hard to model but separates pragmatic from sophisticated |

### Tier 3 — Match context
| Signal | Why |
|---|---|
| Host advantage flag | USA / Canada / Mexico playing at home should not be `neutral=True` |
| Days rest between matches | Fatigue is real, especially in knockouts |
| Travel distance | Big in 2026 WC since teams criss-cross USA / Canada / Mexico |
| Crowd dynamic | Home crowd for hosts vs neutral elsewhere |

### Tier 4 — Modernise the historical inputs
| Change | Why |
|---|---|
| Drop pre-2010 matches from training | Different game; contaminates the model with old patterns |
| Add 2023-2026 matches | 3.5 years of unseen data: 2022 WC + Euros 2024 + Copa America 2024 + Asian Cup 2024 + qualifiers + friendlies |
| Recompute ELO through 2026 | Current ELOs are stale by 3.5 years |

---

## Phase A — Data acquisition

### A1. FBref player stats pipeline
- **What:** Per-player club match stats (goals, assists, minutes, xG, xA, position-specific metrics) for the last 1–2 seasons for every WC squad player (~1,200 players).
- **Source:** [FBref](https://fbref.com) — free, scrapeable, no Cloudflare. Rate-limited to ~10 req/min, so plan accordingly.
- **Output:** `data/processed/player_form_2024_2026.csv` keyed by player+season.

### A2. Recent international lineups
- **What:** Starting XI for every national team's matches in the last ~2 years.
- **Source:** FBref international tab, or Transfermarkt match pages.
- **Output:** `data/processed/intl_lineups.csv` keyed by (date, team, player) with starter/sub flag.

### A3. Coach metadata
- **What:** Current coach + start date for each of 48 WC teams.
- **Source:** Wikipedia (manual or scraped).
- **Output:** `data/processed/coaches.csv` keyed by team.

### A4. Match context features
- **What:** For each 2026 WC match: venue, travel distance from previous match, days rest, crowd composition.
- **Source:** football-data.org schedule + simple distance lookup.
- **Output:** Joined into the feature pipeline.

### A5. Extended match results (2023-2026)
- **What:** All international matches from Nov 2022 to present.
- **Source:** football-data.org for 2024 onward; existing D1/D2 datasets cover earlier.
- **Output:** Extended `matches_clean.csv` with the new window.

### A6. Live injury / availability data (optional)
- **What:** Per-player fit/injured/suspended status updated through tournament.
- **Source:** Twitter feeds, Transfermarkt, official team announcements. Hard. Manual updates may be more practical.
- **Defer if unworkable** — note as Phase 3.1.

---

## Phase B — Feature engineering

### B1. Aggregate player form → team-level
- Weighted by minutes played + position.
- Position-aware: forwards' xG matters differently than defenders' tackles.
- Outputs: `team_form_xg_per_90`, `team_form_goals_per_90`, `team_form_clean_sheet_rate`, etc.

### B2. Chemistry features
- `same_club_concentration` — share of squad in top 3 clubs.
- `lineup_continuity_2yr` — average shared minutes between current first-choice XI over last 2 years.
- `coach_tenure_days` — days since current coach took over.

### B3. Context features (per-match)
- `is_host_at_home` — binary flag, 1 if team is co-host playing on home soil.
- `days_rest` — days since this team's last match.
- `travel_km` — distance from previous match venue.
- Reuse / extend `neutral_flag` — `neutral=False` for hosts playing at home.

### B4. Refresh ELO through 2026
- Recompute via existing `compute_elo` pipeline on extended match results.
- Output: updated `final_elos.csv`.

### B5. Drop pre-2010 from training set
- Update notebook `04_model_training.ipynb` to filter `date >= '2010-01-01'`.
- Verify match count stays in usable range (~14K matches).

---

## Phase C — Modeling

### C1. Baseline retrain
- Same XGB×3 + RF×1 architecture.
- New features added incrementally — check each one's impact.
- Hyperparams start from Phase 2's best (`n=500`, `lr=0.03`, `depth=5`).

### C2. Validation
- Held-out: international matches from Jan 2024 to May 2026 (~600 matches).
- Metrics: outcome accuracy, RPS, Brier, ECE.
- **Target:** RPS ≤ 0.20, ECE < 0.025, outcome accuracy > 60%.

### C3. Sanity check vs 2026 WC matches already played
- Run model on the WC matches we have results for.
- Compare predictions to Phase 2's predictions for the same matches.
- Identify if the new features fix the specific misses (USA-Paraguay etc.) without overfitting.

### C4. Compare to baselines
- Random (33%) — sanity floor.
- "Pick higher ELO" — ~50-55%.
- Bookmaker odds (if obtainable) — ~55-60%, gold standard.
- We should beat ELO baseline by a meaningful margin to justify the complexity.

---

## Phase D — Deployment

### D1. Backend swap
- Replace `models/phase2_model.pkl` → `phase3_model.pkl` on HF Spaces.
- Replace `data/processed/team_features_by_year.csv` (now includes player-form-derived features).
- Update `app/core/predictor.py` to load any new feature lookups.

### D2. Regenerate predictions
- Re-run deterministic prediction → new `frontend/src/lib/deterministic-data.ts`.
- Re-run 10K Monte Carlo simulation for tournament winner probabilities.
- **Wholesale replace** all predictions on `/live` — past and upcoming — with Phase 3 output. As long as no 2026 match entered training, this is honest engineering, not hindsight bias.

### D3. Frontend
- Single set of predictions sourced from Phase 3.
- Methodology page section: "What changed in Phase 3 and why" — documents the new features + the dropped historical data window.

### D4. Documentation
- `EXPERIMENT_LOG.md` — Phase 3 entries documenting each feature's contribution.
- `LEARNINGS.md` — what worked, what didn't, what surprised us.
- Methodology page — public-facing summary.

---

## Risks and how we handle them

| Risk | How we handle it |
|---|---|
| FBref scraping fails or gets blocked | Figure it out as it comes. Plan A: Playwright + 1-req-per-10s throttle. Plan B: Transfermarkt for similar coverage. Plan C: StatsBomb (paid). Don't pre-engineer for failures we may not hit. |
| Player form data has gaps for non-EU-league players | Pull from secondary sources for those players (Iranian/Uzbek/Panama domestic leagues). FM23 already in repo is the natural fallback for whatever still slips through. |
| New features hurt calibration | Standard ML hygiene — add one at a time, watch ECE on the held-out set, drop anything that hurts. Same approach Phase 2 used in `EXPERIMENT_LOG.md`. |
| **Hindsight overfitting to 2026 WC matches** | **Non-negotiable: no 2026 match enters the training set, ever.** Validate exclusively on 2024-2025 international matches. Run on 2026 matches only at the end as the final sanity check. If we keep training clean, retroactive predictions on past 2026 matches are honest model output, not hindsight. |
| Mid-tournament deployment looks like cheating | Not a real concern — nobody is auditing this in real time. As long as risk #4 is enforced, the model's predictions are honest no matter when they're computed. |

---

## Portfolio narrative

> I built a football prediction model in three phases.
> - **Phase 1** used results-based features (ELO + Dixon-Coles bivariate Poisson) and hit confederation bias — Mexico, Japan, Australia were inflated in tournament rankings.
> - **Phase 2** added EA FC squad ratings, broke the echo chamber, but couldn't see player-level form or team chemistry. Shipped before the 2026 WC.
> - **Phase 3** added recent club form (FBref), squad chemistry (same-club concentration + lineup continuity), host-nation flag, match context, and trained on the modern era only. Validated against literature benchmarks (RPS ≤ 0.20, ECE < 0.025). Each phase was driven by a specific limitation identified in the previous.

That's a portfolio story that demonstrates engineering judgment, not just modeling.

---

## Resolved / open

| Question | Resolved |
|---|---|
| FBref scraping vs paid Opta | FBref (free, scrapeable, Opta-licensed data underneath) |
| Per-player form vs team aggregate | Per-player → aggregated to team. Both granularities engineered. |
| Live injury data | **Deferred to Phase 3.1.** Hard to source historically, manual upkeep at runtime. |
| Phase 2 predictions kept side-by-side? | **No.** Wholesale replace once Phase 3 ships (clean train/test discipline makes retroactive predictions honest). |

## Next step

**Phase A1 — FBref scraping infrastructure.** Concrete first task:

1. Pick 3 test players covering different leagues:
   - European star — e.g., Vinicius Júnior (Real Madrid)
   - Asian player — e.g., Mehdi Taremi (Olympiacos)
   - CONCACAF player — e.g., Cyle Larin (Southampton)
2. Visit each FBref profile, document the URL pattern + which stat tables we want.
3. Write a single-player function `fetch_fbref_player(url)` that parses one page and returns structured stats (goals, assists, minutes, xG, xA per season).
4. Verify parsed data matches what's visible on the page.
5. *Then* expand to the resolver (map our 1,247 players → FBref URLs) and the throttled batch scrape.

Time budget: the test/parse step is 1-2 hours. Full pipeline once verified: ~2 hours of throttled scraping (1 req per 6 sec × 1,247 players).
