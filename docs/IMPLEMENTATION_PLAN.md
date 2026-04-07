# FIFA 2026 Predictor — Implementation Plan (Phase 1)

## Project Structure

```
fifa-2026-predictor/
├── data/
│   ├── D1/                    # martj42 international results
│   ├── D2/                    # patateriedata daily updates
│   ├── D3/                    # Fjelstul World Cup DB
│   └── processed/             # Cleaned, merged, feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_tournament_simulation.ipynb
├── src/
│   ├── data/                  # Data loading, cleaning, merging scripts
│   ├── features/              # Feature engineering (ELO, form, H2H)
│   ├── models/                # Model training, evaluation, serialization
│   └── simulation/            # Tournament simulation logic
├── autoresearch/              # AutoResearch experiment configs
├── backend/                   # FastAPI app
│   ├── api/
│   ├── models/
│   └── main.py
├── frontend/                  # Next.js app
│   ├── src/
│   └── public/
├── models/                    # Saved trained models (.pkl / .joblib)
├── PRD.md
├── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## Step 1 — Data Preparation

**Goal:** Single clean dataset ready for feature engineering.

### 1.1 Data Exploration (`01_data_exploration.ipynb`)
- Load D1, D2, D3 and inspect shapes, columns, dtypes, nulls
- Check date ranges — confirm both D1 and D2 cover up to March 2026
- Identify overlapping matches between D1 and D2
- Explore D3 structure — understand how tournament metadata links to matches
- Visualize: matches per year, matches per tournament type, score distributions

### 1.2 Data Cleaning & Merging (`02_data_cleaning.ipynb`)
- Standardize team names across D1, D2, D3 (e.g., "Korea Republic" vs "South Korea")
- Merge D1 + D2 — deduplicate on (date, home_team, away_team, score)
- Keep the richer row when duplicates exist (D1 has `city` column, D2 doesn't)
- Map tournament names to categories: `world_cup`, `continental`, `qualifier`, `friendly`, `other`
- Map teams to confederations using D3's `teams.csv`
- Handle missing values (scores, dates, venues)
- Output: `processed/matches_clean.csv` — single source of truth

### 1.3 Qualified Teams for 2026
- Research and hardcode the 48 qualified teams (or as many as confirmed so far)
- Map to group assignments if available
- Output: `processed/teams_2026.csv`

---

## Step 2 — Feature Engineering

**Goal:** For every match in history, compute features that describe both teams at that point in time.

### 2.1 ELO Rating System (`03_feature_engineering.ipynb`)
- Initialize all teams at 1500
- Iterate chronologically through all matches
- For each match:
  - Look up current ELO for both teams
  - Compute expected result (We) using rating difference + home advantage
  - Compute K based on tournament type + goal difference multiplier
  - Update both teams' ratings
- Output: `processed/elo_ratings.csv` — team, date, elo_rating (snapshot after every match)
- Validation: compare our computed ELOs with eloratings.net for a few known teams/dates

### 2.2 Form Features
- For each team at each match date, compute over last N matches (N = 5 and 10):
  - Win rate
  - Goals scored per match
  - Goals conceded per match
  - Goal difference
  - Points per match (W=3, D=1, L=0)
- Only count competitive matches (exclude friendlies option as a toggle)

### 2.3 Head-to-Head Features
- For each matchup (Team A vs Team B), compute historical:
  - Win rate for each team
  - Average goals scored by each team
  - Number of previous meetings
- Window: all-time and last 5 meetings

### 2.4 Tournament Features
- Tournament type (one-hot or ordinal: friendly < qualifier < continental < world_cup)
- Neutral ground flag (binary)
- Confederation for each team (one-hot encoded)

### 2.5 Assemble Final Feature Matrix
- For each match: combine home_elo, away_elo, elo_diff, home_form_5, away_form_5, h2h_win_rate, tournament_type, neutral, confederations
- Target variable: match outcome (home_win / draw / away_win) — 3-class classification
- Secondary target: goal difference (for scoreline prediction — regression)
- Output: `processed/features_matrix.csv`
- Train/test split: time-based (train on pre-2022, test on 2022 WC + recent matches)

---

## Step 3 — Model Training & Comparison

**Goal:** Train 5 models, compare them rigorously, pick the best.

### 3.1 Baseline Models (`04_model_training.ipynb`)
- **Logistic Regression** — multinomial, 3-class (home/draw/away)
- **Random Forest** — default hyperparams first
- **XGBoost** — default hyperparams first
- **Neural Network** — small MLP (2-3 hidden layers)
- **Ensemble** — VotingClassifier (XGBoost + RF + LR)
- Metrics: accuracy, F1 (macro), log loss, confusion matrix
- Cross-validation: time-series split (5 folds, chronological)

### 3.2 AutoResearch Optimization (`autoresearch/`)
- Set up AutoResearch config pointing at the training script
- Define research directions: hyperparameter search, feature subset selection, class weighting
- Let it run overnight — keep changes that beat current best log loss
- Log all experiments (MLflow or W&B)

### 3.3 Model Comparison (`05_model_comparison.ipynb`)
- Comparison table: all models × all metrics
- Plots: accuracy bar chart, confusion matrices, ROC curves, feature importance (for tree models)
- Calibration plot — are predicted probabilities reliable?
- Backtest: simulate 2022 WC using model predictions, compare to actual results
- Select best model for production use

### 3.4 Scoreline Prediction (Stretch)
- Train a separate regression model (or Poisson model) for goals scored by each team
- Use match outcome model + scoreline model together for full predictions

---

## Step 4 — Tournament Simulation

**Goal:** Use the best model to simulate the full FIFA 2026 tournament.

### 4.1 Simulation Engine (`06_tournament_simulation.ipynb` + `src/simulation/`)
- Input: 48 teams, group assignments, model
- Group stage: predict all group matches → compute standings (points, GD, tiebreakers)
- Knockout stage: predict each match sequentially through the bracket
- Handle draws in knockout: use model's win probability to decide (no draws in knockouts)
- Run simulation N times (e.g., 10,000) with probability-weighted random outcomes
- Output per team: win probability, expected stage reached, group exit probability

### 4.2 Outputs
- `processed/predictions/tournament_winner_probs.json` — probability for each team
- `processed/predictions/group_standings.json` — predicted standings per group
- `processed/predictions/bracket.json` — most likely knockout bracket
- `processed/predictions/team_profiles.json` — per-team stats (ELO, form, predictions)

---

## Step 5 — Backend (FastAPI)

**Goal:** API that serves predictions and supports the UI.

### 5.1 Endpoints
```
GET  /api/predictions/winner          → tournament winner probabilities
GET  /api/predictions/groups          → group stage standings
GET  /api/predictions/bracket         → full knockout bracket
POST /api/predictions/head-to-head    → predict any Team A vs Team B
GET  /api/models/comparison           → model metrics and comparison data
GET  /api/teams                       → list of all 48 teams with stats
GET  /api/teams/{team_id}             → single team profile
```

### 5.2 Architecture
- Load pre-computed predictions from JSON files (fast, no live inference needed for Phase 1)
- Head-to-head endpoint: load trained model, compute features on the fly, predict
- Serve model comparison data (metrics, charts as JSON)
- CORS enabled for Next.js frontend

---

## Step 6 — Frontend (Next.js)

**Goal:** Clean, interactive UI that makes predictions explorable.

### 6.1 Pages
| Page | What it shows |
|---|---|
| `/` (Home) | Tournament winner probability chart (top 10-15 teams), hero section |
| `/groups` | 12 group tables with predicted standings, click to expand match predictions |
| `/bracket` | Interactive knockout bracket — visual tree from R32 to final |
| `/predict` | Head-to-head predictor — pick two teams, see W/D/L probabilities + predicted score |
| `/models` | Model comparison — accuracy table, charts, feature importance, 2022 backtest results |
| `/about` | How it works, methodology, links to GitHub notebooks |

### 6.2 Components
- Probability bar chart (Recharts)
- Group table component
- Bracket tree visualization
- Team selector dropdown
- Match prediction card
- Model metrics table

### 6.3 Design
- Dark theme (football/sports aesthetic)
- Responsive (mobile-friendly for LinkedIn shares)
- Fast — all data pre-fetched, minimal API calls

---

## Execution Order

| # | Task | Depends on | Output |
|---|---|---|---|
| 1.1 | Data exploration | Data downloaded | Understanding of data structure |
| 1.2 | Data cleaning & merging | 1.1 | `processed/matches_clean.csv` |
| 1.3 | 2026 qualified teams | — | `processed/teams_2026.csv` |
| 2.1 | ELO computation | 1.2 | `processed/elo_ratings.csv` |
| 2.2 | Form features | 1.2 | Form columns |
| 2.3 | H2H features | 1.2 | H2H columns |
| 2.4 | Tournament features | 1.2 | Tournament columns |
| 2.5 | Assemble feature matrix | 2.1–2.4 | `processed/features_matrix.csv` |
| 3.1 | Train baseline models | 2.5 | 5 trained models |
| 3.2 | AutoResearch optimization | 3.1 | Optimized models |
| 3.3 | Model comparison | 3.1, 3.2 | Best model selected |
| 3.4 | Scoreline prediction | 2.5 | Goals model |
| 4.1 | Tournament simulation | 3.3, 1.3 | Prediction JSONs |
| 5.1 | FastAPI backend | 4.1 | API running |
| 6.1 | Next.js frontend | 5.1 | UI live |

Steps 2.1–2.4 can run in parallel. Steps 5 and 6 can be developed in parallel too.
