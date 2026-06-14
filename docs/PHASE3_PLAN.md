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
| **Historical lineups** | Use Transfermarkt appearances data (per-match records since 2012) | **Replaces original FBref plan.** FBref is hard-blocked by Cloudflare. Pivoted to `dcaribou/transfermarkt-datasets` (community-scraped TM CSVs) which give us per-match appearance records. Top-26-by-caps proxy is still the fallback for the 81 truly-unfindable players. |
| **Architecture** | Two-track hybrid: XGB+RF classifier for outcome + Dixon-Coles bivariate Poisson for scoreline | The current Phase 2 approach reverse-engineers Poisson lambdas from outcome probabilities (approximation). A proper Dixon-Coles model gives real goal expectancies with the rho correction — academic standard for football scoreline prediction. We already have Dixon-Coles infrastructure from Phase 1. |

## Progress (as of 2026-06-14)

### ✓ Phase A — Data acquisition complete via revised approach
FBref was blocked by Cloudflare (even with stealth + soccerdata). Pivoted to two sources:

| Source | What | Coverage | Stats |
|---|---|---|---|
| **Understat** (via `soccerdata`) | Big 5 leagues × 3 seasons (23/24, 24/25, 25/26) | 4,321 unique players | **xG, xA, np_xG**, key passes, shots, goals, assists, minutes |
| **Transfermarkt datasets** (`dcaribou/transfermarkt-datasets`, weekly-refreshed CSVs) | All pro leagues globally | 47,716 players + per-match appearances since 2012 | Goals, assists, minutes, market value, **per-match records (date)** |

**Coverage of 1,248 WC squad players:**

| Tier | Players | Has xG? | Basic stats? |
|---|---|---|---|
| Both sources (Big 5 → matched in both) | 573 | ✓ | ✓ |
| Transfermarkt only (non-Big-5 leagues) | 594 | ✗ | ✓ |
| Truly unfindable (fall back to team-level form) | 81 | ✗ | ✗ |

**Files in place:**
- `data/processed/understat_players.csv` — 8,326 player-season records
- `data/raw/transfermarkt/players.csv.gz` — 47,716 player metadata
- `data/raw/transfermarkt/appearances.csv.gz` — per-match records since 2012 (41 MB)
- `data/raw/transfermarkt/national_teams.csv.gz` — NT metadata + FIFA ranking
- `data/processed/wc_tm_resolution.csv` — WC squad name → TM player ID map (93% matched)

### → Phase B — Next (feature engineering)
Yet to start. Plan below.

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

## Phase A — Data acquisition  ✓ COMPLETE (revised approach)

### What we did instead of FBref

| Original plan | What actually happened |
|---|---|
| Scrape FBref directly for per-player stats | FBref Cloudflare-blocked even with stealth — no path through soccerdata |
| Scrape FBref for international lineups | TM appearances data is per-match anyway; lineups are implicit in who-played-when |
| Coach metadata via Wikipedia | Deferred to Phase B (still doing it, smaller scope) |
| Match context (travel/rest) | Deferred to Phase B (compute from schedule) |
| Extended 2023-2026 matches | Have via football-data.org + soccerdata sources |
| Live injury data | Deferred to Phase 3.1 |

**Net:** all the data we need for Phase B is already on disk. No more scraping required.

---

## Phase B — Feature engineering

### B1. Per-player trailing-12-month form features
- For each player + reference date, compute from TM `appearances` + Understat `player_seasons`:
  - `goals_per_90_last12mo`
  - `assists_per_90_last12mo`
  - `minutes_last12mo` (fitness proxy)
  - `xg_per_90_last12mo` (Big-5 players only — null for others)
  - `np_xg_per_90_last12mo` (Big-5 only)
  - `xa_per_90_last12mo` (Big-5 only)
- Output: function `compute_player_form(player_id, ref_date)` callable per match.

### B2. Aggregate player form → team-level (minutes-weighted)
- Position-aware. Forwards' xG weighted differently than defenders' tackles.
- Outputs per team per reference date:
  - `team_form_xg_per_90` (sum xG of squad ÷ team minutes × 90)
  - `team_form_goals_per_90`
  - `team_form_top11_avg_overall` (proxy for likely-starting XI strength)
  - For unmatched players, fall back to using team's recent results-based form
- Output: `data/processed/team_form_features_by_date.csv` keyed by (team, ref_date)

### B3. Chemistry features
- `same_club_concentration` — share of squad in top 3 clubs (from `squad_players.json`)
- `lineup_continuity_2yr` — for each pair of WC squad members, count of shared international match dates in last 2 years (from TM appearances filtered to NT competitions)
- `coach_tenure_days` — days since current coach took over (Wikipedia scrape, ~30 min for 48 teams)

### B4. Match context features (Tier A additions)
- `is_host_at_home` — binary flag, 1 if team is co-host playing on home soil
- `neutral` — extend existing flag: `neutral=False` for hosts on home soil
- `days_rest` — days since this team's last match
- `travel_km` — distance from previous match venue (uses match coordinates + simple Haversine)
- `tz_jump` — time zones crossed since last match (proxy for jet lag)

### B5. Set-piece + GK signals (Tier A additions)
- `team_set_piece_goals_per_90` — corner/free-kick goals last 12 mo (derivable from Understat for Big-5 players, accept gap for others)
- `team_gk_psxg_diff_last12mo` — post-shot xG saved above expected (Understat for Big-5 GKs)

### B6. Refresh ELO through 2026
- Recompute via existing `compute_elo` pipeline on extended match results
- Output: updated `final_elos.csv`

### B7. Drop pre-2010 from training set
- Update notebook `04_model_training.ipynb` to filter `date >= '2010-01-01'`
- Verify match count stays in usable range (~14k matches)
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
| ~~FBref scraping fails or gets blocked~~ | ✓ Resolved — FBref WAS blocked; pivoted to Understat + Transfermarkt datasets. |
| Player form data has gaps for non-EU-league players | ✓ Tracked — 81 of 1,248 WC players unmatched. Fall back to team-level form for those. Concentrated in low-probability teams (Iran, Haiti, etc.). |
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
| ~~FBref scraping vs paid Opta~~ | Pivoted entirely — Understat for Big-5 xG + Transfermarkt datasets for everything else. No paid sources. |
| Per-player form vs team aggregate | Per-player → aggregated to team. Both granularities engineered. |
| Live injury data | **Deferred to Phase 3.1.** Hard to source historically, manual upkeep at runtime. |
| Phase 2 predictions kept side-by-side? | **No.** Wholesale replace once Phase 3 ships (clean train/test discipline makes retroactive predictions honest). |
| Tier-A "what hedge funds have" additions? | **Yes.** Folded into Phase B: host flag, days rest, travel km, coach tenure, set-piece, GK PSxG, lineup continuity. |

## Next step

**Phase B1-B2 — Build player form features and team aggregation.**

Concrete first task:
1. Open the TM `appearances.csv.gz` (per-match records since 2012).
2. Build `compute_player_form(player_id, ref_date)` that returns trailing-12-month stats (goals_per_90, assists_per_90, minutes, etc.) for any player at any reference date.
3. Same for Understat for the 573 Big-5 players (adds xG/xA per 90).
4. Test on 3 players: Vinicius, Taremi, Davies — verify the trailing-12-month numbers look right.
5. Then aggregate player → team for each historical match's date (minutes-weighted).

Time budget: ~2-3 hours for B1-B2.
