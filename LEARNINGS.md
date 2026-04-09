# Learnings — FIFA 2026 Predictor

Deep understanding gained while building this project. Not experiment results (see `EXPERIMENT_LOG.md`) — this is the *why* behind things, conceptual insights, and gotchas we discovered.

---

## 1. ELO: What It Is and What It Isn't

**ELO is a 1960s chess formula adapted for football.** It compresses a team's entire strength into a single number. The formula: `We = 1/(1+10^(-dr/400))` — the 400 constant and base-10 are from chess with no football-specific derivation. K-factors (WC=60, qualifier=40, friendly=20) are hand-tuned.

**What ELO captures well:**
- Relative ordering of teams globally (correlation 0.476 with match outcome — our #1 feature)
- Self-correcting over time — bad results punish, good results reward
- Cross-confederation comparison — a team's ELO reflects results against all opponents, unlike DC which only sees within-confederation

**What ELO fundamentally cannot capture:**
- Attack vs defense separately (Spain: elite attack, average defense — ELO can't tell you)
- WHO played (B-squad in a friendly vs full-strength WC XI both update the same ELO)
- HOW you won (dominant 70% possession 1-0 vs lucky counter-attack 1-0 = same ELO change)
- Squad rebuilds (a team that changed 8 players still carries the old squad's ELO)
- Home advantage granularity (Brazil at Maracanã is not the same as New Zealand at home, but ELO uses a flat +100 for everyone)

**The deeper problem — ELO measures the wrong thing:**

ELO measures: *"How many matches have you won recently and against whom?"*
What we actually need: *"How good is this team RIGHT NOW?"*

These are different questions. ELO is backward-looking — it tells you what a team DID. It can't tell you what a team CAN DO with their current squad. This creates two specific distortions:

- **Confederation inflation/deflation:** Mexico wins 8/10 CONCACAF qualifiers against weak teams → ELO rises to 1987. Brazil wins 5/10 CONMEBOL qualifiers against Argentina, Colombia, Uruguay → ELO only 2085. Mexico looks close to Brazil in ELO, but their squad quality is a tier below. ELO treats all wins equally — it doesn't know that beating Honduras and beating Argentina are fundamentally different achievements.

- **Squad turnover blindness:** A team that replaced 8 players since their last ELO-building run still carries the old squad's rating. ELO has no mechanism to detect that the humans behind the team changed.

This showed up in our simulation: Mexico at 7.74% win probability (4th!), Norway at 4.60% (8th), while Portugal sat at 1.21% and Brazil at 6.67%. The model was fooled by confederation-inflated ELOs.

**Why ELO still works for us:** XGBoost doesn't care that the formula is imperfect — it just sees a number that correlates with outcomes. The model learns non-linear patterns on top of ELO (that's what `elo_diff_sq` captures). We use ELO as a *useful input signal*, not as the ground truth of team strength. But it needs a counterweight — an independent measure of squad talent that doesn't come from match results.

**The real insight:** ELO needs an independent counterweight — a signal not derived from match results. EA FC squad ratings (assessed by scouts per player) break the echo chamber because they measure "how good IS this team" rather than "how did this team DO recently."

---

## 2. Why the Neutral Flag Matters (Even Though World Cup Is All Neutral)

**The question:** If all WC matches are played on neutral ground, why do we need a neutral flag at all?

**The answer:** Training data ≠ prediction data.

- ~60% of our 60K training matches are non-neutral (qualifiers, continental cups, friendlies)
- Home teams win ~46% of all matches vs ~27% for away teams
- The neutral flag tells the model: "this team won because they're good" vs "this team won partly because they were at home"

**What breaks without it:**

1. **ELO gets corrupted.** Without neutral flag, ELO can't distinguish home advantage from skill. A team that won 5 home qualifiers gets inflated ELO — the model thinks they're genuinely stronger when they just had crowd/travel advantage.

2. **Dixon-Coles absorbs home advantage into attack params.** DC has an explicit `home_adv` parameter (1.3x — home teams score 30% more). Remove it, and that 30% boost gets baked into attack parameters. Teams that hosted more matches look like better scorers than they actually are.

3. **Form features become noisy.** `win_rate_5` for a team that played 4 home qualifiers + 1 away game looks amazing. Same team with 4 away + 1 home looks mediocre. Without neutral awareness, the model can't distinguish "good team playing away" from "bad team."

**The analogy:** If you're studying for an exam but can't distinguish between questions you got right on your own vs ones where someone whispered the answer — your self-assessment would be off. You need the truth during training to learn accurately, even if the exam itself is fair (neutral).

**For WC prediction:** Everything is neutral anyway — `neutral=1` at inference. The flag only matters during training/evaluation.

---

## 3. Dixon-Coles: The Goal-Level Model ELO Can't Be

**What DC does differently:** Instead of "who wins?", DC asks "what's the most likely scoreline?" using a bivariate Poisson model with a correction for low-scoring games.

Each team gets an attack param and defense param. Expected goals: `λ = exp(attack_home + defense_away + home_adv)`. The rho correction (ρ = -0.14) makes 0-0 and 1-1 draws more likely than raw Poisson — matching real football where both teams playing conservatively is common.

**Why DC as a direct voter was our 2nd biggest improvement (-0.0135):**
The XGB/RF ensemble can't perfectly reproduce a Poisson distribution from tabular features. Giving DC its own weighted vote in the blend adds genuinely orthogonal signal — it's a different *kind* of model, not just more features for the same kind of model.

**DC's fatal flaw: it can't compare across confederations.**
DC learns attack/defense params from match results. A team that scores 3 goals per game in AFC qualifiers looks as strong as one that scores 3 per game in UEFA Euro. But the opponents are vastly different quality.

This is why DC had Australia and Mexico in the top 5 — they farmed goals against weak opponents in their confederation. ELO handles this correctly because it adjusts for opponent strength globally.

**The solution:** Use each model for what it's good at:
- ELO (via XGB/RF) → global ranking, cross-confederation comparison
- DC → goal distribution, scoreline modeling, within-confederation dynamics
- For log loss evaluation: DC×5 weight (full signal for calibrated probabilities)
- For tournament simulation: DC×1 weight (let ELO-based XGB/RF drive realistic rankings)

---

## 10. The Confederation Echo Chamber: Why Our Simulation Gets Rankings Wrong

**The problem we observed:** Our 10K simulation gives Mexico 7.74% win probability (4th), Norway 4.60% (8th), while Portugal gets 1.21% and Brazil 6.67%. Mexico has never won a World Cup. Norway hasn't qualified since 1998. This is clearly wrong.

**Root cause: both main signals are biased in the same direction.**

Our model has two sources of team strength — and both come from the same underlying data (match results):

1. **ELO** (drives XGB/RF, weight 4+1): Mexico beats Honduras, El Salvador, Jamaica in CONCACAF → ELO rises to 1987. Brazil beats/loses to Argentina, Colombia, Uruguay in CONMEBOL → ELO only 2085. ELO treats all wins equally — it doesn't know that beating Honduras and beating Argentina are fundamentally different achievements.

2. **Dixon-Coles** (direct voter, weight 5): Mexico scores 2.5 goals/match against weak CONCACAF teams → DC gives them a high attack rating. Brazil scores 1.2 goals/match against elite CONMEBOL teams → DC gives them a medium attack rating. DC has zero concept of opponent quality — it just counts goals.

Both signals are inflated by the same thing (weak opponents) and deflated by the same thing (strong opponents). When they agree, the model becomes very confident — confidently wrong.

**Why this is an echo chamber, not independent confirmation:** If two witnesses saw the same crime from different angles, their agreement is meaningful. But if both witnesses are just repeating what they heard from the same source, their agreement means nothing. ELO and DC are both "hearing" from match results against the same opponents. They're not independent — they're correlated noise.

**The fix:** Add a signal that is completely independent of match results. Squad quality assessed by scouts (EA FC ratings) doesn't care how many goals Mexico scored against Honduras. It looks at each player individually: this striker is 78 rated, that one is 92. When aggregated, it gives a team strength measure that breaks the echo chamber.

- ELO says Mexico ≈ Brazil → both high from their respective results
- DC says Mexico ≈ Brazil → both score lots of goals
- Squad rating says Mexico << Brazil → 78 avg vs 85 avg, completely different tier

Now the model has a disagreement to learn from. It discovers: "when ELO/DC say a team is strong but squad rating says they're mediocre, the squad rating is more reliable for World Cup prediction."

---

## 4. Calibration: Why It's the #1 Lever for Log Loss

**The insight:** Log loss doesn't care about accuracy — it cares about *how wrong your probability estimates are*. A model that says "70% home win" and is wrong loses less than one that says "95% home win" and is wrong.

**Tree-based models are naturally overconfident.** XGBoost and Random Forest output "hard" probabilities clustered near 0 and 1 (because leaves contain mostly one class). A leaf with 8 home wins and 2 draws outputs 0.8/0.2/0.0 — even though the true probability might be 0.6/0.25/0.15.

**Isotonic calibration** (wrapping models with `CalibratedClassifierCV(method='isotonic', cv=5)`) was our single most reliable improvement. It directly maps "model says 0.8" to "historically, 0.8 predictions are actually right 0.65 of the time" — smoothing overconfident predictions.

**Gotcha discovered in EXP-10:** Forgetting isotonic calibration when reproducing the baseline inflated log loss from 0.7988 to 0.8381. A 0.04 gap just from missing calibration wrappers — it's that important.

---

## 5. Why Draws Are Nearly Impossible to Predict

**Best draw recall across all models: 0.41 (Logistic Regression).** The ensemble achieves only 0.09 draw recall. This isn't a model failure — it's a data reality.

**Why:**
- Draws are the rarest outcome (~23% of matches) vs home wins (~46%) and away wins (~27%)
- Draws are inherently unpredictable — they require both teams to be roughly equal AND neither team to get lucky. Any tiny edge (one good shot, one defensive error) breaks the draw.
- There are no strong predictive features for draws specifically. `abs_elo_diff` (close ELO = more draws) and `dc_draw_prob` are the best signals, but they're weak.

**The trap:** Draw-specific features (EXP-20) improved F1 but *hurt* log loss. Making the model more confident about draws increases the penalty when it's wrong — and draws are wrong most of the time. Log loss heavily punishes confident incorrect predictions.

**Practical implication:** Accept low draw recall. The model correctly assigns ~23% draw probability to evenly-matched games — it just won't predict "this specific match will be a draw" because that's essentially random.

---

## 6. More Models ≠ Better Ensemble

**Tried and failed:** LightGBM, GradientBoosting, ExtraTrees, stacking, additional voters.

**Why more models hurt:**
- Tree-based models (XGBoost, LightGBM, GBM, ExtraTrees, RF) are all correlated — they learn similar patterns from the same features. Adding another tree model adds computational cost without diversity.
- Stacking (meta-learner on top of base models) needs enough signal from base model outputs to learn meaningful combinations. With only 2-3 base models, the meta-learner doesn't have enough dimensions.
- The sweet spot for diversity is **different model families**: tree-based (XGB/RF) + probabilistic (Dixon-Coles). Adding DC as a voter worked because it's fundamentally different — Poisson-based, not tree-based.

**Rule of thumb:** Only add a model to the ensemble if it's a genuinely different *kind* of model, not just a variant of what you already have.

---

## 7. Opponent-Adjusted Features: Sounds Good, Doesn't Help Much

**The hypothesis (EXP-10):** Form features treat all opponents equally — beating Moldova 11-1 gives the same `win_rate_5` boost as beating Italy 4-1. Adding `avg_opp_elo`, `elo_weighted_win_rate`, and `performance_vs_expected` should fix this.

**The reality:** 14 new features added, 0 made the top 25 by XGBoost importance. All ranked 31-63 out of 73 features. The improvement was marginal at best.

**Why:** ELO already implicitly captures opponent strength. When you beat a strong team, your ELO rises more. When you beat a weak team, it rises less. So `elo_diff` (our #1 feature) already encodes "how strong is this team relative to their schedule." The new features were redundant with information ELO already provides.

**The lesson:** Before engineering new features, ask: "does ELO (or another existing feature) already capture this signal?" If the answer is "mostly yes," the new feature will have marginal value at best.

---

## 8. Reproducing Results: The Exact Recipe Matters

**Gotcha from EXP-10:** Baseline reproduced at 0.8381 instead of 0.7988 — a 0.04 gap from three "small" implementation differences:

1. **Missing isotonic calibration** — train.py wraps XGB and RF with `CalibratedClassifierCV`. Forgetting this inflates log loss significantly.
2. **DC voter neutral flag hardcoded** — 57% of test matches are non-neutral. Hardcoding `neutral=True` destroyed home advantage signal for 1,890 matches, amplified by DC×5 weight.
3. **sample_weight in XGB.fit()** — train.py does NOT use sample_weight. RF handles class imbalance via `class_weight` param instead.

**The lesson:** When running experiments that compare against a baseline, reproduce the baseline *exactly* first. Verify the number matches before testing anything new. Even "minor" differences in training setup compound into large metric gaps.

---

## 9. DC Cross-Confederation Blindness

**The root cause of inflated predictions for Mexico, Japan, Australia:**

DC learns team strength from goals scored/conceded. AFC qualifiers produce high-scoring matches against weak teams (e.g., Australia 11-0 Samoa). DC can't tell these goals apart from goals scored in UEFA Nations League against France.

**Two separate problems:**
1. **Qualifier volume** — AFC/CONCACAF teams play 10-18 qualifiers vs weak opponents, pumping up attack params
2. **Tournament quality imbalance** — AFCON/Asian Cup weighted same as UEFA Euro in DC fitting, but the competition quality is vastly different

**The fix wasn't in DC itself** — it was in how we *use* DC. Keep DC×5 for log loss (it genuinely helps calibration) but reduce to DC×1 in simulation so ELO-based XGB/RF drives realistic rankings.

---

## 11. Outcome-First Simulation: Why Sampling Order Matters

**The problem with score-first (Phase 1):** Generate Poisson scoreline → derive outcome. When the model says Spain has 65% win probability, raw Poisson sampling doesn't necessarily produce 65% wins. The Poisson lambda pair that "best matches" the probabilities is an approximation — there's always a gap between the grid-fitted lambdas and the model's actual probabilities.

**The fix — outcome-first (Phase 2):**
1. `np.random.choice(['home', 'draw', 'away'], p=[ph, pd, pa])` — outcome decided *directly* by model probabilities
2. Then rejection-sample a Poisson scoreline that matches the chosen outcome

**Why this matters for simulation:**
- Score-first: strong teams sometimes lose because Poisson randomness overrides model confidence. A team with 70% win probability might only win ~62% of Poisson-sampled matches due to approximation error.
- Outcome-first: the 70% team wins exactly ~70% of the time. Scorelines are still varied and realistic (drawn from Poisson), but they're *conditioned* on the correct outcome.

**Impact on results:**
- Top 6 teams now hold 78.6% of win probability (was ~63% with score-first)
- Only 27 teams won the 2022 backtest at least once (was 29) — less chaos, more realistic
- Argentina's 2022 backtest win%: 24.5% (was 23.8%) — marginally up, more consistent

**The key insight:** For Monte Carlo simulation, you want the *dice to be fair* (calibrated). Outcome-first ensures the dice ARE the model's probabilities. Score-first puts an intermediary (Poisson grid approximation) between the model and the dice — adding unnecessary noise.

---

## 12. FIFA-Style R32 Bracket: Why Third-Place Seeding Matters

**The old bracket (Phase 1):** 8 best third-place teams played *each other* — 4 matches of thirds vs thirds. This created an easy path to the semifinals for any team that finished 3rd.

**The problem:** In Phase 1 simulations, France won the tournament while finishing 3rd in their group. Third-place teams faced other thirds in R32, then potentially faced runners-up — never meeting a group winner until the QF or later. A mediocre team could cruise through weak opposition.

**The fix — FIFA-style seeding:**
- 8 matches: group winners vs third-place teams (cross-group, rotated by 4 to avoid same-group collisions)
- 4 matches: remaining group winners vs cross-group runners-up
- 4 matches: runners-up vs runners-up

**Impact:** Third-place teams now face group winners in R32 — the hardest possible opponent. This makes the third-place path genuinely difficult, matching real FIFA tournament design philosophy where finishing lower in your group = harder knockout draw.

---

## 13. EA Squad Ratings: The Independent Signal That Broke the Echo Chamber

**Phase 1's fundamental problem:** Both ELO and Dixon-Coles come from match results. Teams in weak confederations get inflated by both — it's not independent confirmation, it's correlated noise from the same source.

**What EA ratings provide:**
- Scout-assessed player attributes (pace, shooting, passing, defending, physicality) on a 1-99 scale
- Completely independent of match results — a Mexican player rated 78 is 78 whether Mexico beat Honduras 5-0 or not
- 22 squad-level features (avg overall, positional averages, depth, star power, balance) plus 23 engineered difference features = 45 new features

**The 83/17 split:** ELO and EA ratings agree 83% of the time (France > Norway, Brazil > Bolivia). The value is the 17% where they disagree — teams like Mexico and Japan whose ELO is inflated by weak-region wins, but whose player ratings reveal they lack individual quality to compete with Europe and South America's best.

**Phase 2 results (EA + ELO, no DC):**
- Log loss: 0.826 (vs 0.799 with DC) — slightly worse on paper, but simulation rankings are far more realistic
- ECE: 0.018 (vs 0.027) — *better* calibration
- Mexico: 7.74% → 0.60%, Japan: 2.84% → 0.65%, Australia: 1.08% → 0.04%
- France: 11.7% → 18.1%, Brazil: 6.7% → 7.4%, Argentina: 7.5% → 11.1%

**Why we shipped Phase 2 despite higher log loss:** The purpose of this model is tournament simulation, not match-by-match prediction. A model with 0.826 log loss but realistic rankings is more useful than 0.799 with Mexico at 7.74%. The ECE improvement (0.018 < 0.027) confirms Phase 2 is actually *better calibrated* — it just doesn't have DC's Poisson distribution pulling down log loss.

---

*Last updated: 2026-04-10*
*Add new learnings as we discover them. Focus on "why" — experiment results go in EXPERIMENT_LOG.md.*
