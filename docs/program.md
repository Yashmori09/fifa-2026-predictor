# FIFA 2026 Match Outcome Predictor — AutoResearch Program

## Task
Improve a 3-class football match outcome predictor (home_win / draw / away_win).
Minimize **log_loss** on the held-out test set. Lower is better.

## Baseline
- Model: Ensemble (XGBoost + Random Forest + Logistic Regression), soft voting weights [3, 2, 1]
- log_loss: **0.8333**
- accuracy: **0.6212**
- f1_macro: **0.5313**
- Train: 35,304 competitive matches (1884–Nov 2022)
- Test: 3,313 competitive matches (Nov 2022–Mar 2026, includes 2022 WC)

## Metric
The script prints `METRIC: <value>` on the last line. Lower is better (log loss).
A run is an improvement if METRIC is lower than the current best.

## Features (45 total, already engineered)
- ELO (3): home_elo_before, away_elo_before, elo_diff
- Form (20): win_rate, avg_scored, avg_conceded, pts_per_match, matches_played — last 5 and last 10 competitive matches — for both teams
- H2H (5): h2h_home_win_rate, h2h_home_avg_scored, h2h_home_avg_conceded, h2h_total_meetings, h2h_recent_win_rate
- Static (17): neutral, tournament_importance, confederation one-hots (UEFA/CAF/AFC/CONCACAF/CONMEBOL/OFC/UNKNOWN × home/away), same_confederation

## Known weaknesses
1. Draw prediction is the hardest class — all models struggle (F1 ~0.13–0.37 for draws)
2. MLP collapsed to home_win bias — do NOT use MLP as-is
3. XGBoost plateaus around 250 trees — more trees don't help
4. Class imbalance: home_win 54.7%, away_win 24.3%, draw 20.9% — balanced class weights applied

## What to try (experiment ideas — pick one per run)

### Hyperparameter tuning
- XGBoost: try max_depth 3–8, learning_rate 0.01–0.1, subsample 0.6–1.0, colsample_bytree 0.6–1.0
- XGBoost: try reg_alpha 0.0–1.0, reg_lambda 0.5–5.0
- XGBoost: try min_child_weight 1–10 (helps generalization)
- RF: try n_estimators 100–500, max_depth 6–20, min_samples_leaf 1–20
- LR: try C 0.01–10.0, solver lbfgs/saga, penalty l1/l2/elasticnet
- Ensemble weights: try [4,2,1], [3,3,1], [2,2,2], [5,1,1], [3,2,2]

### New models to add to ensemble
- GradientBoostingClassifier (sklearn) — different from XGBoost, may add diversity
- LightGBM (lgb.LGBMClassifier) — faster, often better than XGBoost on tabular
- ExtraTreesClassifier — more random than RF, better diversity
- CalibratedClassifierCV wrapping RF or XGB — improves probability calibration (directly reduces log loss)

### Calibration (directly targets log loss)
- Wrap any model with CalibratedClassifierCV(method='isotonic') or method='sigmoid'
- Apply calibration to XGBoost (it's known to have overconfident probabilities)
- Apply calibration to Random Forest (hard probability outputs near 0/1)

### Ensemble strategy
- Replace VotingClassifier with stacking: use LR as meta-learner on top of XGB+RF predictions
- Try StackingClassifier with passthrough=True (includes original features in meta-layer)
- Try different base estimator combinations: XGB+LR only, XGB+RF only, XGB+GBM+LR

### Feature engineering (add to FEATURE_COLS list)
- elo_diff_squared: captures non-linear dominance effect
- home_form_momentum: home_win_rate_5 - home_win_rate_10 (trending up or down?)
- away_form_momentum: away_win_rate_5 - away_win_rate_10
- elo_form_interaction: elo_diff * home_pts_per_match_5
- h2h_weight: h2h_recent_win_rate * (h2h_total_meetings / (h2h_total_meetings + 5)) — confidence-weighted H2H
- goal_diff_form: (home_avg_scored_5 - home_avg_conceded_5) - (away_avg_scored_5 - away_avg_conceded_5)

### Class imbalance strategies
- Try class_weight = {home_win: 0.5, draw: 2.0, away_win: 1.5} — stronger draw upweighting
- Try SMOTE on training data (from imbalanced-learn) before fitting
- Try threshold optimization: find optimal decision threshold for draws post-training

## Constraints
- Do NOT change the train/test split or load different data files
- Do NOT add features that use future information (ELO is already recorded pre-match)
- Keep all feature engineering inside train.py (do not modify CSV files)
- The script must complete in under 5 minutes
- Always print the three lines: accuracy, f1_macro, log_loss, then METRIC

## File to modify
`train.py` — modify the MODEL DEFINITION section and optionally add feature engineering before it.
The data loading, encoding, and output format must remain unchanged.
