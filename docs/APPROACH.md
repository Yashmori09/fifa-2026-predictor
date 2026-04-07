# FIFA 2026 Predictor — Approach & Design Decisions

This document captures the reasoning behind every major decision in the project — from data selection to preprocessing to model choices. If something was done a certain way, the "why" is here.

---

## 1. Data Selection

### Why three datasets instead of one?

No single dataset covers everything we need. Each fills a specific gap:

| Dataset | What it covers | Why we need it |
|---|---|---|
| D1 (martj42) | 49k+ international matches, 1872–2026 | Broadest coverage — includes friendlies, qualifiers, tournaments. Has `city` column D2 lacks |
| D2 (patateriedata) | 51k+ matches, auto-updated daily | ~2k more matches than D1 — fills gaps, especially recent ones. Acts as a cross-reference to catch missing data |
| D3 (Fjelstul) | World Cup only, 1930–2022, 27 sub-datasets | Deep tournament metadata — squads, cards, substitutions, group standings, referees. No other dataset has this level of WC-specific detail |

**Why not just use the biggest one (D2)?**
D2 lacks the `city` column. D1 has it. Also, merging two independent sources and deduplicating gives us higher confidence that we're not missing matches. D3 is a completely different kind of data — it's not match results, it's tournament structure.

### Why go back to 1872?

We don't train on all of it equally. Older matches are used only for ELO warmup — the rating system needs history to stabilize. For actual model training, we focus on ~1990 onwards (modern football era with current tournament formats). But ELO needs the full timeline to converge to meaningful ratings.

### Why not use FIFA Rankings dataset directly?

We found that FIFA's ranking data has no public API and requires scraping. More importantly, research papers consistently show that **ELO ratings outperform FIFA rankings** as a predictive feature. Since we can compute ELO from our existing match data, we get a stronger feature for free without an extra data dependency.

We may still add FIFA ranking points later as a secondary feature — but ELO is the priority.

---

## 2. Data Preprocessing

### Why merge D1 + D2?

To get the most complete match history possible. They're independently maintained, so each has matches the other missed. We deduplicate on `(date, home_team, away_team, home_score, away_score)` to avoid counting any match twice.

**When duplicates exist:** We keep the D1 row because it has the `city` column. If D1 is missing the match entirely, we take D2's row.

### Why standardize team names?

Different datasets use different names for the same team:
- "Korea Republic" vs "South Korea"
- "Côte d'Ivoire" vs "Ivory Coast"  
- "Czechia" vs "Czech Republic"

D1 includes a `former_names.csv` and D2 includes `countries_names.csv` — we use these to build a canonical name mapping. If we don't do this, the same team gets treated as two different teams in ELO computation and feature engineering — a silent, serious bug.

### Why categorize tournaments?

Our data has 100+ different tournament names ("FIFA World Cup", "Copa América", "UEFA Euro qualification", "Friendly", "FIFA Series", etc.). For two reasons:

1. **ELO K-value:** The ELO formula uses different weights for different tournament types. World Cup = 60, qualifiers = 40, friendlies = 20. We need clean categories to assign K correctly.
2. **Feature engineering:** Tournament type is a feature itself. A team's performance in friendlies means less than in competitive matches. We bucket into: `world_cup`, `continental_final`, `qualifier`, `friendly`, `other`.

### Why time-based train/test split instead of random?

Random splitting would leak future information into the training set. Example: if a 2024 match is in training and a 2023 match is in test, the model has seen the future when predicting the past.

**Our split:** Train on everything before 2022 World Cup. Test on 2022 World Cup + matches after. This simulates the real use case — we train on history and predict a future tournament.

---

## 3. Feature Engineering

### Why ELO over raw win/loss records?

Win/loss doesn't account for opponent strength. Beating Brazil and beating a small island nation count the same in raw stats. ELO adjusts for this — you gain more rating for beating a strong team and lose more for losing to a weak one. It's a single number that encodes a team's strength relative to all other teams, built from their entire match history.

### Why compute ELO from scratch instead of using eloratings.net?

1. **Reproducibility** — anyone can clone our repo and regenerate the same ELO ratings from raw data
2. **Customization** — we can tune K values, home advantage, goal difference multipliers
3. **Time-stamped** — we need ELO at *every point in time*, not just current. Pre-computed ratings only give you the latest snapshot
4. **No external dependency** — eloratings.net could change format, go down, or not update in time

### Why use both ELO and FIFA ranking?

They capture different signals:
- **ELO** — purely results-based, updates after every match, self-correcting
- **FIFA ranking** — includes a decay factor and confederation weighting that ELO doesn't

Using both lets the model decide which signal is more useful (and in what combination). If they're redundant, the model will learn to downweight one — no harm done.

### Why recent form over last 5 AND 10 matches?

5-match form captures hot streaks and immediate momentum. 10-match form captures medium-term trajectory. A team could be on a 5-match winning streak but have a mediocre 10-match record (suggesting the streak may be a fluke) — or vice versa. Giving the model both windows lets it weigh short vs. medium-term trends.

### Why head-to-head as a feature?

Some matchups have historically lopsided records that aren't captured by either team's individual stats. Example: a mid-ranked team might consistently beat a higher-ranked opponent due to stylistic matchup advantages. H2H captures this pair-specific signal.

**Caveat:** H2H is sparse — many team pairs have played very few times. We include `number_of_meetings` as a feature so the model can learn to discount H2H when the sample size is small.

### Why one-hot encode confederations?

Confederation isn't ordinal (you can't say "UEFA > CAF" as a number). One-hot encoding lets the model learn that inter-confederation matchups behave differently. Example: CONMEBOL teams historically overperform in World Cups relative to their rankings.

### Why separate classification (outcome) from regression (scoreline)?

Match outcome (W/D/L) and exact scoreline are fundamentally different prediction tasks:
- **Outcome** is a 3-class classification — well-suited to tree models and logistic regression
- **Scoreline** involves predicting two correlated count variables (goals for each team) — better suited to Poisson regression or separate regressors

Combining them into one model forces compromises. Keeping them separate lets each model focus on what it's good at. In the UI, we show the outcome prediction (from classification) alongside the predicted score (from regression).

---

## 4. Model Selection

### Why these 5 specific models?

| Model | Why included |
|---|---|
| **Logistic Regression** | Every ML project needs a simple baseline. If tree models only beat LR by 1%, maybe the features matter more than the model — that's useful to know. Also fully interpretable (coefficients show feature impact). |
| **Random Forest** | Multiple papers specifically used RF for World Cup prediction with good results. Handles non-linear relationships, robust to outliers, built-in feature importance. |
| **XGBoost** | Consistently the top performer in tabular data competitions and sports prediction papers. Better at finding subtle patterns than RF through sequential boosting. |
| **Neural Network (MLP)** | Portfolio piece — shows you can go beyond classical ML. Also captures complex feature interactions that tree models might miss. Kept small (2-3 layers) because our dataset isn't huge. |
| **Ensemble (Voting)** | Research shows XGBoost + RF voting achieves the highest accuracy. Different models make different mistakes — combining them cancels out individual weaknesses. |

### Why not deep learning (LSTM, Transformer)?

Our data is tabular with engineered features — not sequential raw data. Deep learning shines on images, text, and long sequences. For structured tabular data with <50k rows, gradient boosting consistently outperforms deep learning. Adding LSTM/Transformers would add complexity without improving accuracy.

If we had play-by-play event data or player tracking data, deep learning would make more sense. That's a potential Phase 2 exploration.

### Why AutoResearch for optimization?

Traditional hyperparameter tuning (GridSearch, Optuna) searches a predefined parameter space. AutoResearch goes further — it can modify the code itself, trying things like:
- Different feature subsets
- Alternative preprocessing steps
- Novel feature combinations
- Unusual hyperparameter ranges

It runs autonomously overnight, which means we get hundreds of experiments without manual effort. For a portfolio project, this is also a great talking point — "I let an AI agent run 500 experiments overnight and it found optimizations I wouldn't have tried."

### Why time-series cross-validation?

Standard k-fold CV randomly shuffles data, which leaks future data into training folds. Time-series CV respects chronological order:
- Fold 1: train on 1990–2005, test on 2006–2008
- Fold 2: train on 1990–2008, test on 2009–2011
- Fold 3: train on 1990–2011, test on 2012–2014
- ...and so on

This gives a realistic estimate of how well the model predicts *future* matches — which is exactly what we need for 2026.

---

## 5. Tournament Simulation

### Why Monte Carlo simulation (10,000 runs)?

A single deterministic prediction ("Brazil beats Argentina") ignores uncertainty. The model outputs probabilities, not certainties. Monte Carlo simulation runs the tournament 10,000 times, each time using probability-weighted random outcomes. This gives us:
- **Win probability per team** (not just "who wins" but "how often does each team win")
- **Confidence intervals** (wide intervals = unpredictable, narrow = strong favorite)
- **Expected stage reached** (a team might not win the tournament but consistently reach the semis)

10,000 runs is enough for stable probabilities while keeping computation fast.

### Why handle knockout draws specially?

In real World Cup knockouts, draws go to extra time and penalties — there's always a winner. Our match outcome model predicts W/D/L. In knockout simulation, if the model predicts a draw, we use the win probabilities to decide who advances (essentially: flip a weighted coin using each team's win probability). This avoids infinite loops and reflects that draws in knockouts still produce a winner.

---

## 6. Backend & Frontend Decisions

### Why pre-computed predictions + one live endpoint?

Phase 1 is a static predictor. Re-running inference on every API call is wasteful when the predictions don't change. Pre-computing and serving from JSON files means:
- Instant response times
- No model loading overhead
- Can deploy on any cheap hosting (no GPU needed)

The one exception is head-to-head — users pick arbitrary team pairs, so we can't pre-compute all 48×47 = 2,256 combinations. For this single endpoint, we load the model and compute features on the fly.

### Why FastAPI over Django?

- Lighter weight — we don't need Django's ORM, admin, auth, templating
- Async by default — better for concurrent API requests
- Auto-generated API docs (Swagger) — nice for the portfolio
- Yash already knows FastAPI well from work projects

### Why Next.js over Streamlit?

Streamlit would be faster to build but:
- Looks like a data science demo, not a real product — weaker portfolio piece
- Limited customization for complex UI (bracket visualization, interactive charts)
- Not mobile-friendly out of the box
- Next.js produces a polished, shareable web app that non-technical people can use from LinkedIn

### Why dark theme?

Sports/football aesthetic. Also easier on the eyes for data-heavy dashboards. Most modern sports apps (ESPN, FotMob, OneFootball) use dark or dark-accent themes.
