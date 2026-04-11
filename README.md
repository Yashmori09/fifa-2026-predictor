# FIFA 2026 World Cup Predictor

A machine learning system that predicts FIFA 2026 World Cup match outcomes and simulates the full 104-match tournament using Monte Carlo methods.

**Live:** [fifa-2026-predictor.vercel.app](https://fifa-2026-predictor.vercel.app)

**Log Loss: 0.826** | **ECE: 0.018** (near-perfect calibration) | **97 input features** | **35,304 training matches (1884–2024)**

## How It Works

An ensemble of 4 models votes on every match:

| Model | Copies | What it captures |
|---|---|---|
| **XGBoost** | 3 | Non-linear patterns — ELO gaps matter more in knockouts than group stages |
| **Random Forest** | 1 | 500 independent trees that reduce overfitting and stabilize predictions |

Each match prediction outputs three probabilities (home win, draw, away win). For tournament simulation, we use **outcome-first sampling** — the model's probabilities directly decide the winner, then a Poisson rejection sampler generates a matching scoreline. This ensures simulation results faithfully follow model predictions while producing realistic, varied scores.

The bracket uses the **official FIFA 2026 format** with Annex C third-place allocation (495 combinations) and fixed R16/QF/SF pairings matching FIFA match numbers 73–88.

### Two Ways to See the Result

| Mode | What it does |
|---|---|
| **The Prediction** | Picks the most likely outcome for every match — no randomness. Same answer every time. **Winner: France** |
| **What If?** | One Monte Carlo simulation with random outcomes weighted by model probabilities. Different winner each run. |

### Top Predictions (10,000 simulations)

| Rank | Team | Win Probability |
|---|---|---|
| 1 | Spain | 21.40% |
| 2 | France | 21.03% |
| 3 | Argentina | 10.83% |
| 4 | England | 9.72% |
| 5 | Germany | 9.50% |
| 6 | Brazil | 6.79% |
| 7 | Netherlands | 4.88% |
| 8 | Portugal | 2.42% |
| 9 | Norway | 2.26% |

### 2022 World Cup Backtest
- Predicted champion (Argentina) ranked #2 with 25.3% probability
- Runner-up (France) ranked #3 with 12.6% probability
- Correctly identified top contenders

### The Three Signals

| Signal | Model Importance | What it captures |
|---|---|---|
| **Form, History & Context** | 38.6% | Recent win rate, goals, head-to-head, tournament importance, momentum |
| **EA FC Squad Ratings** | 31.2% | Scout-assessed player attributes (pace, shooting, passing, defending) |
| **ELO Ratings** | 30.2% | Chess-inspired team ratings updated after every international result |

Adding EA squad ratings was critical — it corrects the **confederation bias** where teams like Mexico and Japan get inflated ELO ratings by beating weaker regional opponents. Scout-assessed player ratings break this echo chamber.

## Project Structure

```
fifa-2026-predictor/
├── notebooks/           # Jupyter notebooks (exploration → simulation)
│   └── phase2/          # Phase 2: EA features, backtest, deterministic prediction
├── backend/             # FastAPI backend (predictor + simulator + official FIFA bracket)
│   └── app/core/        # Predictor (97-feature builder) + simulator (Annex C bracket)
├── frontend/            # Next.js, TypeScript, Tailwind CSS (deployed on Vercel)
├── data/                # Raw match data + processed features
├── models/              # Trained .pkl ensemble + metadata
└── research/            # Experiment scripts and analysis
```

## Features (97 total)

- **ELO-based:** ELO difference, squared difference, average, form windows, volatility
- **EA FC Squad Ratings:** Overall, attack, midfield, defense, positional depth, top-player quality
- **Form:** Win/draw/loss rates, goals scored/conceded, streaks (5 and 10 match windows)
- **Historical:** Head-to-head win rates, goals, match count
- **Context:** Neutral flag, tournament tier, confederation matchup features

## Tech Stack

| Layer | Stack |
|---|---|
| **ML & Data** | Python, XGBoost, scikit-learn, NumPy, Pandas, Jupyter |
| **Backend** | FastAPI, Uvicorn, Poisson simulation engine |
| **Frontend** | Next.js, TypeScript, Tailwind CSS |
| **Deployment** | Vercel (frontend), HuggingFace Spaces (backend API) |

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

## Author

**Yash Mori** — AI Engineer

[GitHub](https://github.com/Yashmori09/fifa-2026-predictor) · [LinkedIn](https://www.linkedin.com/in/yash-mori090102/)
