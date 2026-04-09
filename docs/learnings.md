# Learnings

## The Confederation Echo Chamber Problem

**Problem:** Both ELO and Dixon-Coles derive ratings from match results. Teams in weak confederations (CONCACAF, AFC) accumulate inflated ratings by beating weak opponents. Since BOTH signals are match-result-derived, they reinforce the same bias — there's no independent correction.

**Impact:** Mexico ranked 4th (7.74% win prob), Norway 8th (4.60%), while Portugal was 24th (1.21%). Anyone who watches football knows this is wrong.

**Attempted fix:** EA FC / Football Manager player ratings — assessed by scouts, independent of match results. These should break the echo chamber.

**Result:** The fix works *qualitatively* (Brazil > Mexico, Germany > S.Korea) but doesn't improve aggregate log loss. The bias affects too few test matches to move the overall metric.

---

## What We Tried and Why It Didn't Move the Needle

### The Coverage Problem
- EA FC data only exists from 2014 (FIFA 15)
- Our training data goes back to 1884
- Only 17.2% of training matches have squad features for both teams
- XGBoost handles NaN natively, but it still only sees squad signal on 1 in 6 rows

### The Correlation Problem
- ELO and EA squad ratings are 83% correlated
- They agree on most rankings — the model already "knows" France > Norway from ELO
- The 17% disagreement (confederation bias cases) is a small slice of the data
- A small correction on a small slice = negligible aggregate improvement

### The Feature Noise Problem
- 44 home/away squad features (overall, positional, pace, shooting, etc.) add noise when 82% of rows are NaN
- 9 difference features (overall_diff, top3_diff, positional diffs) capture the key signal more cleanly
- Lean always beat full: 68 features > 112 features

### The Data Volume vs Coverage Trade-off
- 35K matches at 17% coverage > 6K matches at 93% coverage
- The pre-2014 matches still teach the model about ELO/form/H2H patterns
- Throwing away 82% of training data for better squad coverage is a bad trade

---

## What Actually Matters for Prediction Quality

1. **ELO diff** — by far the strongest single feature (importance ~0.12)
2. **Dixon-Coles blend** — DC at weight 4-5 is critical, drops log loss by ~0.02
3. **Form features** — recent win rate, goals scored/conceded, momentum
4. **H2H history** — especially recent win rate and meeting count
5. **Tournament importance** — WC matches behave differently than friendlies
6. **Squad quality diff** — marginal improvement (~0.0005) but fixes known bias cases

---

## Technical Lessons

### XGBoost NaN Handling
- XGBoost learns optimal split direction for missing values — don't fillna(0)
- fillna(0) tells the model "this team has rating 0" which is worse than "unknown"
- For ensemble with RF (which can't handle NaN): train separately and manual blend

### Calibration Matters
- Isotonic calibration with CV=5 on XGB and RF
- CalibratedClassifierCV introduces variance between runs (~0.001 log loss)
- This means differences < 0.002 are likely noise

### FM23 Data Pipeline
- FM uses 1-20 attribute scale, position-aware composite needed
- Linear calibration FM→EA: EA = 4.95 * FM + 8.66 (r=0.939)
- Mean diff ±0.01, std 1.86 on 94 overlapping teams — good enough
- Transfer values unreliable (many zeros) — squad_total_value feature is weak for FM teams

### Team Name Matching is Painful
- 5+ naming variants between databases (St Vincent vs Saint Vincent, Curaçao vs Curacao, etc.)
- Year gaps in EA data (team appears in FIFA 19 but not FIFA 15) need forward/backward fill
- CONIFA teams (Sealand, Yorkshire) appear in match data but have no ratings — leave as NaN

---

## Decision: Shipped Phase 2

**Chose Option A:** Ship the fresh Phase 2 model (0.8258 log loss, no DC) over Phase 1 (0.7988).

**Why:** The product is a WC predictor, not a log loss leaderboard. Phase 1 had Mexico at 7.7% to win the World Cup — above Brazil and Portugal. Nobody would take that seriously. Phase 2 has France 20.3%, Argentina 8.9%, Brazil 11.9% — defensible rankings.

**The tradeoff:**
- Log loss: Phase 1 wins (0.7988 vs 0.8258) — Phase 2 is worse at general match prediction
- Calibration: Phase 2 wins (ECE 0.018 vs 0.027) — better calibrated probabilities
- Rankings: Phase 2 wins convincingly — confederation bias eliminated
- Group accuracy (2022 backtest): Phase 1 wins (87.5% vs 62.5%) — but Phase 1's accuracy was partly accidental (DC spreads predictions flat, so upsets aren't punished)
- Champion prediction (2022 backtest): Phase 2 wins (Argentina 23.8% vs 14.8%) — much higher confidence on the actual winner

**Key insight:** Group accuracy is a misleading metric. Phase 1 got 87.5% partly because DC's flat predictions meant it was never confidently wrong — it hedged. Phase 2 correctly says "France should beat Australia" and gets punished when the upset happens, but that's the right prediction to make. High group accuracy from hedging ≠ good model.

---

## Reverse Poisson > Score Override Hack

**Problem:** Phase 1 used DC lambdas for scorelines but model probabilities for outcomes. When they disagreed (DC says away win, model says home win), the score was overridden to match: if model says home win but score is 0-1, force it to 2-1. This creates unrealistic scorelines.

**Solution:** Reverse-engineer Poisson lambdas directly from model probabilities. Given (p_home, p_draw, p_away), find (λ_h, λ_a) whose Poisson-implied outcome probabilities best match. Then sample goals from Poisson(λ_h) and Poisson(λ_a) — the outcome follows naturally from the score.

**Implementation:** Precomputed 40×40 grid of (λ_h, λ_a) → (p_home, p_draw, p_away) via vectorized einsum. Nearest-neighbor lookup is instant. No per-match optimization needed.

**Result:** Scores are always consistent with outcomes. No more "model says home win but DC says away win" conflicts.

---

## Calibration vs Log Loss — Different Things

Phase 2 has **worse log loss but better calibration**. How?

- **Log loss** penalizes every prediction — including the 82% of training data where EA features are NaN noise. Phase 1 + DC is just better at general match prediction across all eras.
- **ECE (calibration)** measures whether "30% predictions happen 30% of the time." Phase 2's isotonic calibration on XGB+RF without DC's interference produces cleaner probability estimates.
- **Bias direction flipped:** Phase 1 overestimated away wins (+0.009) and underestimated home wins (-0.012). Phase 2 does the opposite (+0.013 home, -0.009 away). Removing DC shifted the balance — DC was pulling toward its own confederation-biased away/draw estimates.

**Lesson:** A model can be well-calibrated (probabilities mean what they say) while still having higher log loss (wrong more often on individual predictions). For Monte Carlo simulation, calibration matters more — you want the dice to be fair even if you can't predict individual rolls.

---

## Batch Prediction is Non-Negotiable

Phase 1 backend took 12+ minutes to warm cache (2,256 matchups). Each `predict_proba` call on CalibratedClassifierCV with 500 RF trees = 331ms. One at a time = death.

**Fix:** Build all 2,256 feature vectors into a single numpy matrix, call `predict_proba` once on the full batch. XGB: one call. RF: one call. Blend vectorized.

**Result:** 12 min → 1.7s (420x faster). Same for backtest: 13 min → 49s cache build.

**Lesson:** sklearn models are fast in batch, slow per-sample. Always batch when you have all inputs upfront.
