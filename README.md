# FIFA 2026 World Cup Predictor

A machine learning system that predicts FIFA 2026 World Cup match outcomes and simulates the full 104-match tournament using Monte Carlo methods.

**Live:** [fifa-2026-predictor.vercel.app](https://fifa-2026-predictor.vercel.app)

**Log Loss: 0.804** | **ECE: 0.027** | **62.6% outcome accuracy** | **157 input features** | **6,162 modern-era training matches (2018–2025)**

## How It Works

A hybrid **goal-scoring** model — predicts each team's expected goals (λₕ, λₐ) directly with two Poisson-loss XGBoost regressors, then derives match outcomes from the Dixon-Coles-corrected joint scoreline matrix. This is the same architecture used by FiveThirtyEight's SPI and most serious football models.

| Component | What it does |
|---|---|
| **XGBoost Poisson × 2** | One regressor predicts the home team's expected goals (λₕ), the other the away team's (λₐ). Trained on 6,162 international matches. |
| **Dixon-Coles scoreline** | The 11×11 joint scoreline matrix P(home=i, away=j) is built from the two Poisson distributions and corrected for the four low-score cells (0–0, 1–0, 0–1, 1–1) where real football diverges from independent Poisson. |
| **Outcome derivation** | Above the diagonal = home win, on the diagonal = draw, below = away win. The full scoreline matrix lets the Monte Carlo simulator sample realistic scores for each of the tournament's 103 matches per simulation. |

**Why a goal-scoring model:** a 3-way W/D/L classifier has to learn "what makes a draw" from features. But draws aren't *caused* by features — they happen when two attacking outputs cancel out. Predicting each team's expected goals separately means draws emerge naturally when λₕ ≈ λₐ.

The bracket uses the **official FIFA 2026 format** with Annex C third-place allocation (495 combinations) and fixed R16/QF/SF pairings matching FIFA match numbers 73–88.

### Two Ways to See the Result

| Mode | What it does |
|---|---|
| **The Prediction** | Picks the most likely outcome for every match — no randomness. Same answer every time. **Winner: France** |
| **What If?** | One Monte Carlo simulation with random outcomes weighted by model probabilities. Different winner each run. |

### Top Predictions (100,000 simulations)

| Rank | Team | Win Probability |
|---|---|---|
| 1 | Spain | 13.73% |
| 2 | France | 13.62% |
| 3 | Argentina | 10.28% |
| 4 | England | 5.66% |
| 5 | Brazil | 5.41% |
| 6 | Portugal | 4.74% |
| 7 | Japan | 3.94% |
| 8 | Morocco | 3.81% |
| 9 | Netherlands | 3.59% |
| 10 | Colombia | 3.57% |

No team above 14%. Spread reflects the genuine uncertainty in a 48-team tournament — historical pre-tournament favorites have rarely been priced above 18%.

### The Three Signals

| Signal | Model Importance | What it captures |
|---|---|---|
| **ELO, Form & Match Context** | 42.3% | Chess-inspired team ratings, last-5/last-10 form, head-to-head, confederation, tournament importance, Dixon-Coles attack/defense |
| **EA FC Squad Ratings** | 31.3% | Scout-assessed player attributes per year (FIFA 15 through FC 26) — overall, positional splits, depth, six core team attributes |
| **StatsBomb Intl & Chemistry** | 26.3% | xG profile from 314 international tournament matches (WC 2018/2022, Euro 2020/2024, Copa America 2024, AFCON 2023), recent intl form, squad chemistry (same-club concentration, intl caps, squad age) |

The new StatsBomb-derived features carry 26% of model attention despite being only 16% of the feature count — they're concentrated signal, especially on tournament matches where their training distribution matches the prediction context.

### Validation

Evaluated on **748 international matches from June 2025 to March 2026** — the 12 months immediately before the World Cup. The model never saw these matches during training, and **no WC 2026 match enters either training or validation.**

| Metric | Phase 3 | Squad-rating baseline | ELO baseline |
|---|---|---|---|
| Log Loss | **0.804** | 0.977 | 0.893 |
| Accuracy | **62.6%** | 53.5% | 62.2% |
| RPS | **0.156** | 0.204 | 0.176 |
| ECE | **0.027** | 0.065 | 0.048 |

Slicing by ELO difference:
- **Blowouts (>400 ELO gap):** 91% accuracy — model crushes obvious matches
- **Lopsided (200–400):** 70% accuracy — most R32/R16 matches
- **Close (<50):** 31% accuracy — football's irreducible noise

## Known Limits (the honest part)

**The draw wall.** Football has ~22% draws but every calibrated model predicts draws as the modal outcome only 3–9% of the time. We tested 5 architectures (XGB classifier, hybrid Poisson, ordinal XGB, mord proportional-odds, TabNet). Every approach that solved draw recall blew up the log loss. The tradeoff is structural to the sport, not a defect of this model.

**Close-match collapse.** Below 50 ELO difference (e.g. Spain vs France), accuracy drops to ~31% — essentially random. These are also the matches most likely to be draws.

**Scoreline accuracy.** Exact scoreline prediction lands at 10.8% — tied with naive baselines like "always 1-0." Football scorelines are extremely concentrated (top 6 scorelines account for 53% of all matches). Within-1-goal accuracy is 60% (71% on tournament matches).

## What I Discovered

- **Found a Dixon-Coles leak.** Earlier-iteration DC features were fitted on data through the test window, inflating metrics. Refitting on pre-holdout data only changed dc_home_win_prob >2% for 629/748 matches — the honest log loss is 0.804, not the inflated number we'd have reported.
- **More data wins, not cleaner data.** Aggressive filtering (drop weak teams, drop friendlies, modern era only) hurt every metric. Andorra-Liechtenstein qualifiers still teach the model about Poisson goal distributions.
- **The draw wall is real.** Confirmed across 3 independent approaches (TabNet, aggressive filter, Mord).
- **StatsBomb data carries real signal.** New B5 features account for 26% of model attention; help most on tournament matches specifically (+0.6% accuracy on continental finals).

Full write-up: [docs/PHASE3_PLAN.md](docs/PHASE3_PLAN.md)

## Project Structure

```
fifa-2026-predictor/
├── notebooks/           # Phase 1/2 Jupyter notebooks (exploration → simulation)
├── scripts/             # Phase 3: feature engineering, training, experiments, diagnostics
│   ├── build_*.py       # Feature builders (player form, team form, chemistry, statsbomb, intl)
│   ├── phase3_*.py      # Phase 3 training, hybrid, diagnostics, scoreline eval, simulation
│   ├── exp*.py          # 5 experiments (data filtering, time-varying, ordinal, etc.)
│   └── refit_dc_no_leak.py  # The DC leakage fix
├── backend/             # FastAPI backend (predictor + simulator + official FIFA bracket)
│   └── app/core/        # Phase 3 hybrid predictor + tournament simulator
├── frontend/            # Next.js, TypeScript, Tailwind CSS (deployed on Vercel)
├── data/                # Match data + StatsBomb event data + processed features
├── models/              # phase3_hybrid_clean.pkl + dc_params_v2.json
└── docs/                # PHASE3_PLAN, methodology, learnings, approach, features
```

## Features (157 total)

- **Base (59):** ELO, form-5/10, H2H, confederations, neutral flag, tournament importance, Dixon-Coles
- **Squad (53):** Per-year FIFA / EA FC ratings + 9 difference features
- **StatsBomb International (12):** xG for/against per match, defensive overperformance, attacking overperformance, set-piece share, n_matches
- **International Form (14):** Goals for/against per match, win/draw rate, last-10 form, n_matches in last 2 years, competitive percentage
- **Chemistry (10):** Same-club concentration, unique clubs, avg international caps, avg squad age
- **B5 difference features (9):** Diffs between home and away for the most predictive of the above

## Tech Stack

| Layer | Stack |
|---|---|
| **ML & Data** | Python, XGBoost, scikit-learn, SciPy, NumPy, Pandas, Jupyter |
| **Backend** | FastAPI, Uvicorn, Dixon-Coles + Poisson scoreline engine, HuggingFace Spaces |
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS, Vercel |
| **Data sources** | International match database, ELO ratings, EA FC 26, Football Manager 23, StatsBomb Open Data |

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

**Yash Mori** — AI/ML Engineer

[GitHub](https://github.com/Yashmori09/fifa-2026-predictor) · [LinkedIn](https://www.linkedin.com/in/yash-mori090102/)
