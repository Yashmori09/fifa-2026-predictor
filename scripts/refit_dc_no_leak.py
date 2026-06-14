"""
Refit Dixon-Coles using ONLY matches before the holdout window, then recompute
the per-match dc_* features so the test set has no leakage.

The existing dc_ratings.csv was fit through 2026-03-31, contaminating the test
holdout (2025-06-11 → 2026-03-31). This script:

  1. Fits DC on matches with date < HOLDOUT_START (2025-06-11)
  2. Recomputes dc_lambda, dc_mu, dc_home_win_prob, dc_draw_prob,
     dc_away_win_prob, dc_total_goals, dc_goal_diff for all matches in
     train_dc.csv + test_dc.csv using the leakage-free DC
  3. Saves train_dc_v2.csv + test_dc_v2.csv

Now Phase 3 evaluation is honest:
  • train matches before 2025-06-11 → features reflect a model fit on data
    "as of 2025-06" (consistent across train)
  • test matches in 2025-06 → 2026-03 → features reflect a model that has
    NEVER seen those matches (true held-out evaluation)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"

HOLDOUT_START = pd.Timestamp("2025-06-11")
XI = 0.002          # time-decay used in Phase 2
ANCHOR_TEAM = "England"


def fit_dc(home_idx, away_idx, home_goals, away_goals, weights, n_teams, anchor_idx):
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    def nll(params):
        atk = params[:n_teams]
        dfn = params[n_teams:2 * n_teams]
        ha = params[2 * n_teams]
        rh = params[2 * n_teams + 1]
        lam = np.exp(atk[home_idx] + dfn[away_idx] + ha)
        mu = np.exp(atk[away_idx] + dfn[home_idx])
        tau = np.ones(len(home_goals))
        tau[m00] = 1 - lam[m00] * mu[m00] * rh
        tau[m01] = 1 + lam[m01] * rh
        tau[m10] = 1 + mu[m10] * rh
        tau[m11] = 1 - rh
        valid = tau > 0
        ll = np.sum(weights[valid] * (
            np.log(tau[valid])
            + poisson.logpmf(home_goals[valid], lam[valid])
            + poisson.logpmf(away_goals[valid], mu[valid])
        ))
        return -ll

    init = np.zeros(2 * n_teams + 2)
    init[2 * n_teams] = 0.3
    init[2 * n_teams + 1] = -0.1
    bounds = [(None, None)] * (2 * n_teams + 2)
    bounds[anchor_idx] = (0.0, 0.0)

    res = minimize(nll, init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-6})
    atk = res.x[:n_teams]
    dfn = res.x[n_teams:2 * n_teams]
    ha = float(res.x[2 * n_teams])
    rh = float(res.x[2 * n_teams + 1])
    return atk, dfn, ha, rh, res.success


def compute_match_dc_features(home, away, neutral, atk, dfn, team_idx, home_adv, rho, max_goals=10):
    """Per-match DC features used as inputs to the ML ensemble."""
    if home not in team_idx or away not in team_idx:
        return {"dc_home_win_prob": 0.30, "dc_draw_prob": 0.25, "dc_away_win_prob": 0.45,
                "dc_lambda": 1.4, "dc_mu": 1.1, "dc_total_goals": 2.5, "dc_goal_diff": 0.3}
    hi = team_idx[home]
    ai = team_idx[away]
    ha = home_adv if not neutral else 0.0
    lam = float(np.exp(atk[hi] + dfn[ai] + ha))
    mu = float(np.exp(atk[ai] + dfn[hi]))
    goals = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    M[0, 0] *= max(1 - lam * mu * rho, 1e-10)
    M[0, 1] *= max(1 + lam * rho, 1e-10)
    M[1, 0] *= max(1 + mu * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    hw = float(np.sum(M * (goals[:, None] > goals[None, :])))
    dr = float(np.sum(M * (goals[:, None] == goals[None, :])))
    aw = max(0.0, 1.0 - hw - dr)
    return {
        "dc_home_win_prob": hw,
        "dc_draw_prob": dr,
        "dc_away_win_prob": aw,
        "dc_lambda": lam,
        "dc_mu": mu,
        "dc_total_goals": lam + mu,
        "dc_goal_diff": lam - mu,
    }


def main():
    print("=" * 80)
    print("Refit Dixon-Coles with no leakage into 2025-06 holdout")
    print("=" * 80)

    mc = pd.read_csv(PROC / "matches_clean.csv", parse_dates=["date"])
    mc = mc[mc["home_score"].notna() & mc["away_score"].notna()].copy()
    mc["home_score"] = mc["home_score"].astype(int)
    mc["away_score"] = mc["away_score"].astype(int)

    fit_pool = mc[mc["date"] < HOLDOUT_START].copy()
    print(f"\n[Fit] matches available: {len(fit_pool):,} "
          f"({fit_pool['date'].min().date()} → {fit_pool['date'].max().date()})")

    fit_pool["days_ago"] = (fit_pool["date"].max() - fit_pool["date"]).dt.days
    fit_pool["weight"] = np.exp(-XI * fit_pool["days_ago"])

    all_teams = sorted(set(fit_pool["home_team"]) | set(fit_pool["away_team"]))
    team_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    anchor_idx = team_idx.get(ANCHOR_TEAM, 0)
    print(f"  teams: {n_teams}, anchor: {ANCHOR_TEAM} (idx {anchor_idx})")

    home_idx_arr = fit_pool["home_team"].map(team_idx).values
    away_idx_arr = fit_pool["away_team"].map(team_idx).values
    home_goals_arr = fit_pool["home_score"].values
    away_goals_arr = fit_pool["away_score"].values
    weights_arr = fit_pool["weight"].values

    print("\n[Optimize] L-BFGS-B (~30-60s)...")
    t0 = time.time()
    atk, dfn, home_adv, rho, ok = fit_dc(
        home_idx_arr, away_idx_arr, home_goals_arr, away_goals_arr,
        weights_arr, n_teams, anchor_idx
    )
    print(f"  converged={ok}  time={time.time()-t0:.1f}s")
    print(f"  home_adv={home_adv:.4f}, rho={rho:.4f}")
    print(f"  attack range: [{atk.min():.3f}, {atk.max():.3f}]")
    print(f"  defense range: [{dfn.min():.3f}, {dfn.max():.3f}]")

    # ── Save the leakage-free DC ratings + params
    dc_ratings_v2 = pd.DataFrame({"team": all_teams, "attack": atk, "defense": dfn})
    dc_ratings_v2["overall"] = dc_ratings_v2["attack"] - dc_ratings_v2["defense"]
    dc_ratings_v2 = dc_ratings_v2.sort_values("overall", ascending=False)
    dc_ratings_v2.to_csv(PROC / "dc_ratings_v2.csv", index=False)

    dc_params_v2 = {
        "home_adv": home_adv, "rho": rho, "xi": XI,
        "reference_date": str(fit_pool["date"].max().date()),
        "n_teams": n_teams,
        "fit_window_end": HOLDOUT_START.strftime("%Y-%m-%d"),
        "fit_pool_size": len(fit_pool),
    }
    with open(MODELS / "dc_params_v2.json", "w") as f:
        json.dump(dc_params_v2, f, indent=2)

    print("\n  ✓ saved: dc_ratings_v2.csv, dc_params_v2.json")
    print("\n[Sanity] top 10 by overall (should look reasonable):")
    print(dc_ratings_v2.head(10).to_string(index=False))

    # ── Recompute per-match dc_* features
    print("\n[Recompute] per-match dc_* features for train_dc.csv + test_dc.csv...")
    train = pd.read_csv(PROC / "train_dc.csv", parse_dates=["date"])
    test = pd.read_csv(PROC / "test_dc.csv", parse_dates=["date"])

    def attach_dc(df, label):
        print(f"  {label}: {len(df):,} matches...")
        rows = []
        for _, r in df.iterrows():
            neutral = bool(r.get("neutral.1", r.get("neutral", True)))
            rows.append(compute_match_dc_features(
                r["home_team"], r["away_team"], neutral,
                atk, dfn, team_idx, home_adv, rho,
            ))
        new = pd.DataFrame(rows)
        # Replace existing dc_* columns
        for col in new.columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = pd.concat([df.reset_index(drop=True), new], axis=1)
        return df

    train_v2 = attach_dc(train, "train")
    test_v2 = attach_dc(test, "test")

    train_v2.to_csv(PROC / "train_dc_v2.csv", index=False)
    test_v2.to_csv(PROC / "test_dc_v2.csv", index=False)
    print(f"\n  ✓ saved: train_dc_v2.csv ({len(train_v2):,}), test_dc_v2.csv ({len(test_v2):,})")

    # Quick sanity check: do the new dc_home_win_prob values differ from old ones?
    test_orig = pd.read_csv(PROC / "test_dc.csv")
    holdout_orig = test_orig[pd.to_datetime(test_orig["date"]) >= HOLDOUT_START]
    holdout_v2 = test_v2[test_v2["date"] >= HOLDOUT_START]
    print(f"\n[Diff] dc_home_win_prob change in holdout window:")
    if len(holdout_orig) == len(holdout_v2):
        delta = (holdout_v2["dc_home_win_prob"].values - holdout_orig["dc_home_win_prob"].values)
        print(f"  mean Δ = {delta.mean():+.4f}, max|Δ| = {np.abs(delta).max():.4f}")
        print(f"  → meaningful: {(np.abs(delta) > 0.02).sum()}/{len(delta)} matches shifted by >2%")


if __name__ == "__main__":
    main()
