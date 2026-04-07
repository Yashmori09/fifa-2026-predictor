# FIFA 2026 Predictor — Product Requirements Document

## Overview

A personal portfolio project to predict the FIFA World Cup 2026 winner and simulate the full tournament. Built with real ML experimentation (multiple models, comparisons, AutoResearch), a clean public-facing web app, and a well-documented GitHub repo showcasing the full ML workflow.

**Goal:** Share on LinkedIn. Public users can explore predictions, simulate brackets, and see how the models work.

---

## Tech Stack

| Layer | Tech |
|---|---|
| ML / Experiments | Python, Jupyter Notebooks, scikit-learn, XGBoost, AutoResearch (Karpathy) |
| Model Tracking | MLflow or Weights & Biases |
| Visualization | Matplotlib, Seaborn (notebooks) |
| Backend | FastAPI |
| Frontend | Next.js |
| UI Charts | Recharts (or similar) |
| Heavy training | Google Colab (if needed) |

---

## Phase 1 — Static Predictor

> Data frozen. Model trained once on historical data. Full shippable product.

### Features

- **Tournament Winner Prediction** — probability distribution across all 48 teams
- **Group Stage Simulation** — predict standings for all 12 groups
- **Full Bracket Simulation** — simulate the entire knockout bracket end-to-end
- **Match Outcome Prediction** — Win / Draw / Loss + predicted scoreline for any matchup
- **Model Comparison** — train multiple models (Logistic Regression, Random Forest, XGBoost, Neural Net), compare accuracy, show results with graphs
- **AutoResearch Integration** — use Karpathy's AutoResearch to run autonomous overnight experiments and surface the best-performing model config
- **Visualization Notebooks** — graphs for model performance, feature importance, historical trends — published on GitHub

### UI (Next.js)
- Homepage with tournament winner probabilities (bar/donut chart)
- Group stage table view
- Interactive bracket simulator
- Head-to-head match predictor (pick any two teams)
- Model comparison page (show metrics, charts)

---

## Phase 2 — Live & Advanced

> Builds on Phase 1 without rewriting. Activates closer to / during the tournament (June–July 2026).

### Features

- **Live Updates** — retrain/update predictions as qualifiers finalize, squads are announced, and matches are played during the tournament
- **Player-Level Features** — incorporate player ratings (FIFA ratings or similar), key player absences, injury impact
- **Form-Based Weighting** — recent match results weighted more heavily as tournament approaches
- **"What If" Simulator** — swap players, change team form, replay the bracket with custom inputs
- **Upset / Dark Horse Alerts** — highlight statistically surprising predictions
- **Odds Comparison** — compare model predictions vs. betting market odds as a benchmark

### UI Additions
- Live prediction dashboard (updates as tournament progresses)
- What-if simulator panel
- Dark horse / upset highlight cards

---

## Data Sources

### Phase 1 (Required)

| Dataset | Source | What it gives us |
|---|---|---|
| International football results 1872–2026 | [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | ~50k matches, all internationals including World Cup — **core training data** |
| International Football Results: Daily Updates | [Kaggle](https://www.kaggle.com/datasets/patateriedata/all-international-football-results) | Auto-updated version, good for staying current |
| FIFA World Cup 1930–2022 | [Kaggle](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup) | Tournament-only match data |
| Fjelstul World Cup Database | [GitHub](https://github.com/jfjelstul/worldcup) | Most comprehensive WC dataset — 27 datasets, 1.58M data points (squads, goals, cards, referees) |
| FIFA Rankings History | [GitHub Scraper](https://github.com/cnc8/fifa-world-ranking) | Ranking points over time — strong predictive feature, exports as CSV |

### Phase 2 (Player-level & Live)

| Dataset | Source | What it gives us |
|---|---|---|
| FIFA 24 Player Stats | [Kaggle](https://www.kaggle.com/datasets/rehandl23/fifa-24-player-stats-dataset) | 18,500+ players, 30+ skill metrics (crossing, finishing, dribbling, stamina, etc.) |
| FIFA 23 Player Stats | [Kaggle](https://www.kaggle.com/datasets/bryanb/fifa-player-stats-database) | Backup / historical player ratings |
| API-Football | [api-football.com](https://www.api-football.com/) | Real-time scores, lineups, live stats |
| WorldCupAPI | [worldcupapi.com](https://worldcupapi.com/) | Built specifically for FIFA 2026 — live scores, historical data |

---

## Feature Set (Phase 1)

Confirmed via research paper survey — these features are consistently the strongest predictors across multiple studies.

| Feature | Source | Notes |
|---|---|---|
| ELO Rating | Computed from D1/D2 | Strongest predictor — outperforms FIFA ranking. Compute from scratch using historical match results |
| FIFA Ranking Points | FIFA Rankings dataset | Standard baseline, good complementary signal |
| Recent Form (last 5–10 matches) | Computed from D1/D2 | Win rate, goals scored/conceded in recent window |
| Head-to-Head Record | Computed from D1/D2 | Historical record between specific team pairs |
| Goal Difference | Computed from D1/D2 | Cumulative and recent — better signal than just W/L |
| Neutral Ground Flag | D1/D2 (column exists) | Home advantage is a real factor |
| Tournament Type Weight | D1/D2 (tournament column) | Friendlies are noise — weight WC/qualifiers higher |
| Confederation | D3 (teams.csv) | Regional strength baseline (UEFA, CONMEBOL stronger historically) |

### ELO Rating Computation

We compute ELO from scratch using historical match results. Formula from [eloratings.net](https://www.eloratings.net/about):

- **Base:** `Rn = Ro + K × (W - We)`
  - `Rn` = new rating, `Ro` = old rating
  - `W` = result (1 = win, 0.5 = draw, 0 = loss)
  - `We` = expected result: `1 / (10^(-dr/400) + 1)` where `dr` = rating diff + 100 (home advantage)
- **K values by tournament type:**
  - World Cup finals: 60
  - Continental championship finals: 50
  - WC/continental qualifiers: 40
  - Other tournaments: 30
  - Friendlies: 20
- **Goal difference multiplier on K:**
  - 2-goal win: K × 1.5
  - 3-goal win: K × 1.75
  - 4+ goal win: K × (1.75 + (N-3)/8)
- **Initial rating:** 1500 for all teams

---

## Models to Train & Compare

| Model | Role |
|---|---|
| Logistic Regression | Baseline — simple, interpretable |
| Random Forest | Strong for WC prediction per research |
| XGBoost / Gradient Boosting | Consistently top performer across papers |
| Neural Network (small) | Compare deep learning vs. ensemble |
| Ensemble (XGBoost + RF voting) | Highest accuracy in literature |

AutoResearch will handle hyperparameter tuning and config optimization across these models automatically.

---

## Research Tasks (Before Implementation)

- [x] Find the best available datasets (historical World Cup results, international match results, FIFA rankings, player stats)
- [x] Survey research papers on football/soccer match outcome prediction
- [x] Decide on feature set (team-level for Phase 1, player-level deferred to Phase 2)
- [x] Research ELO rating computation methodology
- [ ] Evaluate AutoResearch setup requirements (GPU, experiment loop config)
- [ ] Identify data gaps and fallback sources

---

## Success Criteria

- Phase 1 live and shareable on LinkedIn before FIFA 2026 starts (June 2026)
- GitHub repo with clean notebooks, model comparison graphs, and documented experiments
- Public users can interact with the predictor without any login
