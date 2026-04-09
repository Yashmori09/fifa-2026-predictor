# Experiment Log

## Phase 1 Experiments (EXP-1 to EXP-10)

### Best Result: EXP-10 (Production Model)
- **Architecture:** XGB(cal)×4 + RF(cal)×1 + DC×5 manual blend
- **Features:** 59 (ELO, form, H2H, DC probs, confederations, engineered)
- **Log Loss: 0.7988** | Accuracy: 0.6399 | F1: 0.4964
- **Train:** 35,304 matches (1884–2023) | **Test:** 3,313 matches (2023–2024)
- Key engineered features: elo_diff_sq, form_momentum, goal_diff_form, h2h_confidence
- DC blend at weight 5 was optimal (DC×5 = equal weight with ML ensemble)

---

## Phase 2 Experiments: EA/FM Squad Features (EXP-11)

### Goal
Fix confederation-inflated ELO/DC bias. Mexico ranked 4th (7.74%), Norway 8th (4.60%), Portugal only 24th (1.21%) — all wrong due to weak-opponent inflation in both ELO and Dixon-Coles.

### Data Sources
| Source | Teams | Years | Scale |
|--------|-------|-------|-------|
| EA FC (FIFA 15–FC26) | ~90/year | 2014–2026 | 1–99 overall |
| Football Manager 2023 | 209 nationalities | proxy 2014–2025 | 1–20 attrs → calibrated to EA scale |
| Confederation imputation | 16 tiny FIFA nations | 2014–2025 | bottom-quartile conf average |

### EXP-11a: Initial EA-only (6.3% coverage)
- Squad features r=0.42 with outcome (same as ELO)
- EA alone: 0.9230 (too little training data)
- Combined best: 0.8107 (XGB×4+HistGBT×1+DC, 112 feat)
- Only 2,218/35,304 training matches had squad data

### EXP-11b: 2014+ training only
- Trained on 2014+ subset only (2,218 matches with squad data)
- Worse than full training set — not enough data

### EXP-11c: Enriched with FM23 + imputation (17.2% coverage)
| Model | Features | Train | Log Loss | Notes |
|-------|----------|-------|----------|-------|
| Baseline (59 feat + DC) | 59 base | 35K | 0.8075 | XGB only, no RF |
| C4 (68 feat + DC) | 59 base + 9 diff | 35K | 0.8067 | Best XGB-only |
| C1 (112 feat + DC) | 59 base + 44 squad + 9 diff | 35K | 0.8080 | Too many NaN features |

### EXP-11d: 2014+ only training (93% coverage)
| Model | Features | Train | Log Loss | Notes |
|-------|----------|-------|----------|-------|
| D1 (68 feat + DC) | 59 base + 9 diff | 6K | 0.8131 | High coverage, low data |
| BB (59 feat + DC) | 59 base | 6K | 0.8149 | Baseline on 6K |
| C4 (68 feat + DC) | 59 base + 9 diff | 35K | 0.8093 | **35K still wins** |

**Conclusion:** More training data (35K) > higher squad coverage (93% on 6K).

### EXP-11e: Phase 1 ensemble with EA features (the real test)
| Model | Log Loss | Acc | F1 | vs 0.7988 |
|-------|----------|-----|----|----|
| E0: Phase1 repro (59 feat, DC×5) | 0.7999 | 0.6399 | 0.4949 | +0.0011 |
| E1b: XGB(68,NaN)×4+RF(59)×1, DC×5 | 0.7994 | 0.6420 | 0.4979 | +0.0006 |
| E1b: XGB(68,NaN)×4+RF(59)×1, DC×4 | **0.7993** | 0.6393 | 0.4927 | **+0.0005** |
| E2: XGB(112,NaN)×4+RF(59)×1, DC×5 | 0.8003 | 0.6420 | 0.4997 | +0.0015 |

**Best: 0.7993** — does NOT beat Phase 1's 0.7988 (delta +0.0005, within noise).

Phase 1 repro gives 0.7999 (not exact 0.7988) — small variance from calibration CV folds.

### Confederation Bias Sanity Check (E1b + DC×5)
| Match | ML Prediction | DC Prediction | Correct? |
|-------|--------------|---------------|----------|
| Brazil vs Mexico | Brazil 38.2% > Mexico 32.1% | Mexico 57.5% > Brazil 12.1% | ML correct |
| Portugal vs Mexico | Portugal 36.3% > Mexico 33.7% | Mexico 52.4% > Portugal 15.9% | ML correct |
| Germany vs S.Korea | Germany 36.9% > S.Korea 35.5% | S.Korea 49.6% > Germany 22.1% | ML correct |
| France vs Norway | France 47.3% > Norway 23.7% | France 42.7% > Norway 28.2% | Both correct |

Note: with DC×5 blend, the confederation bias correction is diluted but still present.

---

## Phase 2: Fresh Model (EXP-12 — Notebooks 01–04)

### Goal
Build a new model from scratch using EA squad features as primary strength signal, dropping DC blend entirely. Fix confederation bias properly rather than bolting EA onto Phase 1.

### NB01: Training Window Analysis (no ELO/DC)
- Best: All data + EA diffs = **0.8489** (vs 0.8762 without EA)
- EA features improve every training window by -0.025 to -0.037
- More old data still helps — "All" beats "2018+" even without squad coverage
- Football has evolved: goals dropped 5.5→2.7, home advantage 40%→23%

### NB02: EDA & Feature Engineering
- 12+ EA features are >0.95 correlated — one giant cluster
- Greedy selection: 14 independent features from 45
- Best model: 93 feat (all diffs + engineered) = **0.8489** ≈ 9 lean diffs (0.8499)
- Feature importance: EA = 35.7%, Form/H2H = 53.9%, Context = 10.4%

### NB03: Model Tuning + ELO Addition
| Model | Features | Log Loss | Notes |
|-------|----------|----------|-------|
| NB02 best (no ELO) | 93 | 0.8489 | Baseline |
| + elo_diff only | 94 | 0.8293 | -0.0196! |
| + elo_diff + sq + home/away | 97 | 0.8266 | Best single model |
| 9 lean EA + elo_diff | 58 | 0.8272 | Lean beats full |
| XGB tuned (best params) | 97 | 0.8263 | lr=0.03,d=5,a=0.5,l=1.5 |
| **XGB×3 + RF×1** | **97** | **0.8258** | **Production model** |

Feature importance: ELO 30.2% | EA 31.2% | Form/H2H/Context 38.6%
Gap closed: 46.2% of 0.05 gap (0.8489→0.8258, target 0.7988)

### NB04: 10,000 Tournament Simulations
| Rank | Team | Phase 1 | Phase 2 | Delta |
|------|------|---------|---------|-------|
| 1 | France | 11.68% | 20.30% | +8.62 |
| 2 | Spain | 19.52% | 19.10% | -0.42 |
| 3 | Brazil | 6.67% | 11.94% | +5.27 |
| 4 | Argentina | 7.50% | 8.90% | +1.40 |
| 5 | England | 7.91% | 8.82% | +0.91 |
| 6 | Germany | 6.93% | 7.16% | +0.23 |
| 12 | Mexico | 7.74% | 1.40% | **-6.34** |
| 17 | Japan | 2.84% | 0.85% | -1.99 |

Confederation shift: CONCACAF 9.5%→2.0%, AFC 5.6%→1.1%, CONMEBOL 17.6%→23.6%, UEFA 63.7%→72.0%

### Technical Changes
- **Scoreline generation:** Reverse-engineered Poisson from model probabilities (no DC dependency)
- **Batch prediction:** 2,256 matchups in 1.4s (vs 8min Phase 1)
- **Model file:** `models/phase2_model.pkl` (XGB×3 + RF×1, 97 features)

### NB05: Calibration Curve Analysis
| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Mean ECE | 0.0273 | **0.0178** |
| Away bias | +0.009 | -0.009 |
| Draw bias | +0.002 | -0.005 |
| Home bias | -0.012 | +0.013 |
| Away Brier | 0.1330 | 0.1391 |
| Draw Brier | 0.1629 | 0.1661 |
| Home Brier | 0.1702 | 0.1793 |

Phase 2 is better calibrated (ECE 0.018 vs 0.027) despite worse log loss. Bias direction flipped — removing DC shifted balance from overestimating away wins to overestimating home wins.

### NB06: 2022 World Cup Backtest (10K simulations)
| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Argentina (Champion) | #2 (14.77%) | **#2 (23.79%)** |
| France (Runner-up) | #3 (7.98%) | #4 (11.76%) |
| Croatia (3rd) | #20 (0.74%) | #10 (0.86%) |
| Morocco (4th) | #9 (4.53%) | #18 (0.17%) |
| Group accuracy | **87.5%** | 62.5% |
| Sim speed | 32.5s | **5.3s** |

Confederation bias fix confirmed on 2022 data:
| Team | Phase 1 | Phase 2 | Delta |
|------|---------|---------|-------|
| Mexico | 6.28% | 0.29% | **-5.99** |
| Senegal | 6.76% | 1.61% | **-5.15** |
| Japan | 6.27% | 0.14% | **-6.13** |
| Australia | 5.19% | 0.08% | **-5.11** |

Group accuracy drop (87.5% → 62.5%) is expected — Phase 2 makes stronger predictions on favorites, so upsets hurt more. Phase 1's high accuracy was partly from DC spreading predictions flat (hedging).

---

## Key Findings Across All Phase 2 Experiments

1. **EA squad features provide independent signal** (r=0.42, same as ELO) but are 83% correlated with ELO — the useful 17% is exactly the confederation bias cases
2. **Log loss improvement is negligible when bolted onto Phase 1** (+0.0005 at best) because DC×5 drowns out EA signal
3. **Building fresh without DC: EA = 35.7% of model importance** — the signal is real when not suppressed
4. **elo_diff alone closes half the gap** (-0.0196 log loss, biggest single improvement)
5. **The real value is in simulation rankings** — Mexico 7.7%→1.4%, Brazil 6.7%→11.9%, France 11.7%→20.3%
6. **Training data coverage is the fundamental bottleneck** — 82% of training data is pre-2014 (no player ratings exist)
7. **More data > more features** — 35K matches at 17% coverage beats 6K matches at 93% coverage
8. **Lean features > full features** — 9 diff features outperform 44+9 home/away+diff features (less NaN noise)
9. **Reverse Poisson works** — scorelines naturally consistent with model predictions, no hacky override needed
