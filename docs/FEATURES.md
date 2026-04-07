# Feature Documentation — FIFA 2026 Predictor

This document explains every feature used in the model — what it means, why we use it, and how it's computed.

---

## What Are We Predicting?

**Target:** `outcome` — one of three classes:
- `home_win` — the first-listed team wins
- `draw` — match ends level
- `away_win` — the second-listed team wins

The model outputs a **probability for each class**, e.g. `[home_win: 65%, draw: 20%, away_win: 15%]`. These probabilities drive the Monte Carlo tournament simulation.

We predict home/away/draw rather than "which country wins" because:
- We have 200+ countries — predicting country names directly means 200+ output classes with very sparse data per class
- Our approach learns **general patterns** (strong ELO gap → favorite wins) that transfer to any matchup, even teams that have never met
- Country identity is irrelevant — the features tell the whole story

---

## Feature Groups

### 1. ELO Rating (3 features)

| Feature | Description |
|---|---|
| `home_elo_before` | ELO rating of home team before this match |
| `away_elo_before` | ELO rating of away team before this match |
| `elo_diff` | home_elo_before − away_elo_before |

**What is ELO?**
A single number representing overall team strength, updated after every match. Every team starts at 1500. The formula:

```
New ELO = Old ELO + K × GDM × (W − We)

K   = tournament weight (World Cup=60, Qualifier=40, Friendly=20)
GDM = goal difference multiplier (bigger wins → bigger ELO change)
W   = actual result (1=win, 0.5=draw, 0=loss)
We  = expected result based on ELO gap (1 / (1 + 10^(-dr/400)))
```

Home advantage: +100 added to home team ELO on non-neutral ground.

**Why it's the most important feature:**
ELO diff has a correlation of 0.476 with match outcome — highest of any single feature. When Spain (ELO ~2270) plays Qatar (ELO ~1400), the 870-point gap tells the model almost everything it needs to know before looking at anything else.

**Current top teams (as of training cutoff):**
Spain ~2270, Argentina ~2227, France ~2200, Brazil ~2190

**Why ELO over FIFA rankings?**
FIFA rankings use a similar formula but are updated less frequently and have political weighting. ELO is recalculated from scratch on every match — more responsive to actual performance.

---

### 2. Recent Form (20 features)

Computed over **last 5 matches** and **last 10 matches** for both home and away team. Friendlies are excluded — only competitive matches count.

| Feature | Description |
|---|---|
| `home_win_rate_5` | Fraction of last 5 competitive matches won by home team |
| `home_avg_scored_5` | Average goals scored per match (last 5) |
| `home_avg_conceded_5` | Average goals conceded per match (last 5) |
| `home_pts_per_match_5` | Points per match (win=3, draw=1, loss=0) over last 5 |
| `home_matches_played_5` | How many of the last 5 slots are filled (new teams have fewer) |
| *(same 5 features for away team)* | |
| *(same 5 features for home team, 10-match window)* | |
| *(same 5 features for away team, 10-match window)* | |

**Why two windows (5 and 10)?**
- Last 5 captures current hot/cold streak — high signal, small sample
- Last 10 captures medium-term momentum — more stable, less noisy
- Together they let the model assess whether a team is trending up or declining

**Why exclude friendlies?**
Teams rotate squads, experiment tactically, and don't play at full intensity in friendlies. Including them would dilute the signal from competitive matches that actually matter.

---

### 3. Head-to-Head (5 features)

Tracks the historical record between these two specific teams directionally.

| Feature | Description |
|---|---|
| `h2h_home_win_rate` | Fraction of historical meetings won by home team |
| `h2h_home_avg_scored` | Average goals scored by home team in H2H meetings |
| `h2h_home_avg_conceded` | Average goals conceded by home team in H2H meetings |
| `h2h_total_meetings` | Total historical meetings between these two teams |
| `h2h_recent_win_rate` | Win rate in H2H weighted toward recent meetings |

**Why H2H matters beyond ELO:**
Some teams consistently over- or under-perform against specific opponents regardless of their ELO. Tactical matchups, psychological factors, and playing styles create patterns that persist over time.

Examples:
- South Korea has historically outperformed their ELO against European sides
- Certain African teams have strong records against South American opponents

H2H correlation with outcome: 0.343 — second strongest after ELO diff.

**Reliability:** H2H features are more meaningful when `h2h_total_meetings` is high. Our `h2h_confidence` engineered feature weights recent win rate by meeting count to handle sparse H2H data.

---

### 4. Match Context (2 features)

| Feature | Description |
|---|---|
| `neutral.1` | 1 if match is on neutral ground, 0 if home team has home advantage |
| `tournament_importance` | Numeric weight of tournament type |

**Tournament importance mapping:**
| Tournament Type | Value |
|---|---|
| World Cup | 60 |
| Continental Championship (Euros, Copa América, etc.) | 50 |
| Qualifying | 40 |
| Other Competitive | 30 |
| Friendly | 20 |

This mirrors the ELO K-value — more important tournaments are worth more, and teams are more motivated.

**Neutral ground:** All 2026 World Cup matches are on neutral ground (USA/Canada/Mexico). Setting `neutral=1` removes the +100 home ELO advantage from the calculation, which is the correct treatment for tournament matches.

---

### 5. Confederation (15 features)

One-hot encoded confederation membership for both teams, plus same-confederation flag.

| Feature | Description |
|---|---|
| `home_conf_UEFA` | Home team is UEFA member (Europe) |
| `home_conf_CAF` | Home team is CAF member (Africa) |
| `home_conf_AFC` | Home team is AFC member (Asia) |
| `home_conf_CONCACAF` | Home team is CONCACAF member (North/Central America) |
| `home_conf_CONMEBOL` | Home team is CONMEBOL member (South America) |
| `home_conf_OFC` | Home team is OFC member (Oceania) |
| `home_conf_UNKNOWN` | Home team confederation unknown |
| *(same 7 for away team)* | |
| `same_confederation` | Both teams from the same confederation |

**Why confederation matters:**
- Playing styles and tactical approaches differ systematically by region
- CONMEBOL and UEFA teams have stronger historical records at World Cups
- Same-confederation matches tend to be more tactical and lower-scoring (familiarity with opponent style)
- African (CAF) teams have historically underperformed their ELO at World Cups — the model learns this

---

### 6. Engineered Features (7 features)

Derived from the base features to capture non-linear patterns and interactions.

| Feature | Formula | Why |
|---|---|---|
| `elo_diff_sq` | `elo_diff² × sign(elo_diff)` | Captures non-linear dominance — a 500-point ELO gap is disproportionately more decisive than a 250-point gap |
| `home_form_momentum` | `home_win_rate_5 − home_win_rate_10` | Positive = team trending up; Negative = team declining. Catches teams that ELO hasn't caught up with yet |
| `away_form_momentum` | `away_win_rate_5 − away_win_rate_10` | Same for away team |
| `home_goal_diff_form` | `home_avg_scored_5 − home_avg_conceded_5` | Net goal dominance in recent matches |
| `away_goal_diff_form` | `away_avg_scored_5 − away_avg_conceded_5` | Net goal dominance for away team |
| `net_goal_diff` | `home_goal_diff_form − away_goal_diff_form` | Relative goal dominance: positive = home team scoring more and conceding less recently |
| `h2h_confidence` | `h2h_recent_win_rate × (meetings / (meetings + 5))` | Confidence-weighted H2H — dampens H2H signal when teams have only met 1-2 times |

---

## Feature Importance Summary

From XGBoost feature importance and correlation analysis:

| Rank | Feature | Insight |
|---|---|---|
| 1 | `elo_diff` | Single strongest predictor (corr: 0.476) |
| 2 | `h2h_home_win_rate` | Historical matchup record |
| 3 | `home_conf_CAF` / `home_conf_UNKNOWN` | Confederation effects are real |
| 4-8 | H2H features cluster | Head-to-head matters beyond ELO |
| 9-15 | Form features | Recent momentum adds signal |
| 16+ | Confederation one-hots | Useful but secondary |

---

## What We Don't Have (Known Limitations)

| Missing Data | Why It Matters | Why We Don't Have It |
|---|---|---|
| Player availability / injuries | "Is Mbappé playing?" is huge signal | Only known at match time — Phase 2 feature |
| Squad strength (FIFA ratings) | Brazil's 23-man squad vs Cameroon's | Data collection across 342 teams × 60k matches is impractical |
| Expected Goals (xG) | Better than actual goals for measuring team quality | Expensive, not freely available for all international matches |
| Betting odds | Best single predictor — encodes professional models | Phase 2 only (only available close to match time) |
| Weather / altitude | Affects high-altitude venues (Mexico City) | Not standardized across historical dataset |

---

## Target Metric: Why Log Loss?

We optimize **log loss** rather than accuracy because:

- **Accuracy** only checks if the top predicted class was correct
- **Log loss** penalizes wrong confidence — a model that says "99% Brazil wins" and Brazil loses is punished far more than one that says "60% Brazil wins"

For a tournament simulator running 10,000 Monte Carlo simulations, we need **calibrated probabilities** — not just correct labels. A model saying "Brazil 65% win, Draw 20%, opponent 15%" produces realistic simulations. A model saying "Brazil 99%" produces the same winner every time regardless of opponent.

Current best log loss: **0.8263** (AutoResearch optimized ensemble)
