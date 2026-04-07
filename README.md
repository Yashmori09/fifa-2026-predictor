# FIFA 2026 World Cup Predictor

A machine learning system that predicts FIFA 2026 World Cup match outcomes and simulates the full tournament bracket using Monte Carlo methods.

**Best log loss: 0.7988** | **ECE: 0.027** (well-calibrated)

## How It Works

Three models vote on every match prediction:

| Model | Weight | What it captures |
|---|---|---|
| **XGBoost** | 4 | Non-linear patterns from 59 features (ELO, form, Dixon-Coles ratings) |
| **Random Forest** | 1 | Ensemble diversity, different decision boundaries |
| **Dixon-Coles** | 5 (eval) / 1 (sim) | Goal distributions via bivariate Poisson — attack/defense per team |

All tree models wrapped with **isotonic calibration** to prevent overconfident probabilities (critical for log loss).

Tournament simulation runs 10,000 Monte Carlo iterations through all 104 matches (12 groups + knockout rounds), producing win probabilities for each team.

### Top Predictions

| Rank | Team | Win Probability |
|---|---|---|
| 1 | Spain | 19.4% |
| 2 | France | 11.2% |
| 3 | Argentina | 8.6% |
| 4 | England | 8.1% |
| 5 | Germany | 7.0% |
| 6 | Brazil | 6.8% |

### 2022 World Cup Backtest
- Predicted champion (Argentina) ranked #2
- 87.5% group stage accuracy

## Project Structure

```
├── notebooks/           # 10 Jupyter notebooks (exploration → simulation)
│   ├── 01-04            # Data exploration, cleaning, feature engineering, training
│   ├── 06               # Dixon-Coles feature integration
│   ├── 07               # Tournament simulation (Monte Carlo)
│   ├── 08               # 2022 WC backtest
│   ├── 09               # Calibration curve analysis
│   └── 10               # Opponent-adjusted features (experiment)
├── src/
│   └── train.py         # Production training pipeline
├── backend/             # FastAPI — /predict and /simulate/full endpoints
├── frontend/            # Next.js — interactive tournament bracket + H2H predictor
├── data/                # Raw match data (D1 + D2 + D3 datasets, gitignored)
├── models/              # Trained .pkl files (gitignored) + metadata .json
├── scripts/             # Experiment helper scripts
├── docs/
│   └── APPROACH.md      # Detailed design decisions
├── EXPERIMENT_LOG.md    # All 30 experiments with results and deltas
└── LEARNINGS.md         # Deep insights (ELO limitations, calibration, DC tradeoffs)
```

## Features (59 total)

- **ELO-based** (6): elo_diff, elo_diff_sq, avg_elo, elo_form_5/10, elo_volatility
- **Form** (20): win/draw/loss rates, goals scored/conceded, streaks (5 and 10 match windows, both teams)
- **Historical** (6): h2h win rates, h2h goals, h2h match count
- **Dixon-Coles** (7): attack/defense ratings, DC win/draw/loss probs, expected goals
- **Context** (4): neutral flag, tournament tier, confederation matchup features

## Key Experiment Results

| Experiment | Change | Log Loss | Delta |
|---|---|---|---|
| Baseline | Initial ensemble | 0.8333 | — |
| Calibration + features | AutoResearch pipeline | 0.8263 | -0.0070 |
| Dixon-Coles features | 7 DC features added | 0.8131 | -0.0132 |
| Hyperparameter tuning | XGB grid search | 0.8123 | -0.0008 |
| **DC as direct voter** | **DC×5 in blend** | **0.7988** | **-0.0135** |

Full experiment log with 30 experiments in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Tech Stack

**ML Pipeline:** Python, scikit-learn (1.2.1), XGBoost, NumPy, Pandas, Jupyter

**Backend:** FastAPI, Uvicorn — loads trained .pkl models, serves predictions

**Frontend:** Next.js 14, TypeScript, Tailwind CSS — interactive bracket visualization with animation controls

## Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev    # http://localhost:3000
```

### Training (reproduce from scratch)
```bash
# Requires data/ directory with D1, D2, D3 datasets
# Notebooks 01-04 build features → src/train.py trains final model
python src/train.py
```

## Documentation

- [APPROACH.md](docs/APPROACH.md) — Design decisions: why three datasets, ELO vs FIFA rankings, Dixon-Coles integration, calibration strategy
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — All experiments with hypotheses, results, and decisions
- [LEARNINGS.md](LEARNINGS.md) — Deep technical insights discovered during development

## Author

**Yash Mori** — [GitHub](https://github.com/Yashmori09) | [LinkedIn](https://linkedin.com/in/YashMori)
