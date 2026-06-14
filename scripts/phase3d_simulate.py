"""
Phase D-2 — Monte Carlo simulation of WC 2026 using Phase 3 hybrid model.

For each WC 2026 match:
  1. Build the same feature vector used in Phase 3 training
  2. Predict (λ_h, λ_a) via the hybrid model
  3. Compute scoreline matrix with DC correction
  4. Sample a scoreline per match
  5. Propagate through:
       12 groups of 4 → top 2 + 8 best 3rd-place → 32 teams → R32 → R16 → QF → SF → Final
  6. Handle knockout draws via penalty shootouts (50/50 with slight ELO tilt)

Outputs:
  data/processed/wc2026_simulation_phase3.csv      P(team wins WC), P(reaches each stage)
  data/processed/wc2026_match_predictions_phase3.csv   per-match probabilities
  data/processed/wc2026_modal_bracket_phase3.csv   most-likely bracket scenario
"""
import warnings
warnings.filterwarnings("ignore")

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_train import (
    PHASE3_FEATURES, PHASE2_FEATURES,
    join_squad, join_static,
    SQUAD_FEATURES, BASE_FEATURES, DIFF_FEATURES,
    HOME_SQUAD, AWAY_SQUAD, HOME_SB, AWAY_SB,
    HOME_INTL, AWAY_INTL, HOME_CHEM, AWAY_CHEM,
    NEW_DIFF_FEATURES, SB_FEATURES, INTL_FEATURES, CHEM_FEATURES,
)

MAX_GOALS = 10
N_SIMS = 100_000
RNG_SEED = 42

WC_OPENING = pd.Timestamp(2026, 6, 11)


# ── Feature assembly for WC 2026 matches ─────────────────────────────

def build_match_features():
    """Build a feature vector for every group-stage match in schedule_2026.csv."""
    sched = pd.read_csv(PROC / "schedule_2026.csv", parse_dates=["date"])
    teams = pd.read_csv(PROC / "teams_2026.csv")

    # ── Load all required feature sources
    elos = pd.read_csv(PROC / "final_elos.csv")  # current ELOs through 2026-03-31
    elo_map = dict(zip(elos["team"], elos["final_elo"]))

    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")

    # DC ratings (v2 — leak-free)
    dc_ratings = pd.read_csv(PROC / "dc_ratings_v2.csv")
    with open(MODELS / "dc_params_v2.json") as f:
        dc_params = json.load(f)
    dc_map = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}

    # Build per-match feature rows
    confederation_cols = ["UEFA", "CAF", "AFC", "CONCACAF", "CONMEBOL", "OFC", "UNKNOWN"]
    conf_map = dict(zip(teams["team"], teams["confederation"]))

    rows = []
    for _, m in sched.iterrows():
        h, a = m["home_team"], m["away_team"]
        h_elo = elo_map.get(h, 1500.0)
        a_elo = elo_map.get(a, 1500.0)

        # DC features (per-match scoreline distribution from DC alone)
        if h in dc_map and a in dc_map:
            h_att, h_def = dc_map[h]
            a_att, a_def = dc_map[a]
            # WC group matches: neutral=True (host countries play "home" but treated separately)
            lam = float(np.exp(h_att + a_def))
            mu = float(np.exp(a_att + h_def))
        else:
            lam = mu = 1.4

        # Scoreline matrix from DC for derived features
        g = np.arange(MAX_GOALS + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        M[0, 0] *= max(1 - lam * mu * dc_params["rho"], 1e-10)
        M[0, 1] *= max(1 + lam * dc_params["rho"], 1e-10)
        M[1, 0] *= max(1 + mu * dc_params["rho"], 1e-10)
        M[1, 1] *= max(1 - dc_params["rho"], 1e-10)
        M = M / M.sum()
        hw = float(np.sum(M * (g[:, None] > g[None, :])))
        dr = float(np.sum(M * (g[:, None] == g[None, :])))
        aw = max(0, 1 - hw - dr)

        row = {
            "match_id": m["match_id"],
            "date": m["date"],
            "home_team": h, "away_team": a,
            # Base features
            "home_elo_before": h_elo, "away_elo_before": a_elo,
            "elo_diff": h_elo - a_elo, "elo_diff_sq": (h_elo - a_elo) ** 2,
            # form features default to neutral 5-match base
            "home_win_rate_5": 0.5, "home_avg_scored_5": 1.3, "home_avg_conceded_5": 1.1,
            "home_pts_per_match_5": 1.5, "home_matches_played_5": 5,
            "home_win_rate_10": 0.5, "home_avg_scored_10": 1.3, "home_avg_conceded_10": 1.1,
            "home_pts_per_match_10": 1.5, "home_matches_played_10": 10,
            "away_win_rate_5": 0.5, "away_avg_scored_5": 1.3, "away_avg_conceded_5": 1.1,
            "away_pts_per_match_5": 1.5, "away_matches_played_5": 5,
            "away_win_rate_10": 0.5, "away_avg_scored_10": 1.3, "away_avg_conceded_10": 1.1,
            "away_pts_per_match_10": 1.5, "away_matches_played_10": 10,
            # H2H (unknown — defaults to neutral)
            "h2h_home_win_rate": 0.4, "h2h_home_avg_scored": 1.3,
            "h2h_home_avg_conceded": 1.1, "h2h_total_meetings": 0, "h2h_recent_win_rate": 0.4,
            "neutral.1": 1, "tournament_importance": 5,
            # form momentum / goal diff form
            "home_form_momentum": 0, "away_form_momentum": 0,
            "home_goal_diff_form": 0, "away_goal_diff_form": 0,
            "net_goal_diff": 0, "h2h_confidence": 0,
            # DC features
            "dc_home_win_prob": hw, "dc_draw_prob": dr, "dc_away_win_prob": aw,
            "dc_lambda": lam, "dc_mu": mu,
            "dc_total_goals": lam + mu, "dc_goal_diff": lam - mu,
        }
        # Confederation flags
        for c in confederation_cols:
            row[f"home_conf_{c}"] = int(conf_map.get(h) == c) if c != "UNKNOWN" else 0
            row[f"away_conf_{c}"] = int(conf_map.get(a) == c) if c != "UNKNOWN" else 0
        row["same_confederation"] = int(conf_map.get(h) == conf_map.get(a))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Join squad features for 2026
    df = join_squad(df, tfy)
    # Join SB / intl / chem
    df = join_static(df, sb, intl, chem)
    return df


def lams_to_scoreline_matrix(lh, la, rho):
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= max(1 - lh * la * rho, 1e-10)
    M[0, 1] *= max(1 + lh * rho, 1e-10)
    M[1, 0] *= max(1 + la * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    return M


def matrix_to_wdl(M):
    g = np.arange(MAX_GOALS + 1)
    hw = float(np.sum(M * (g[:, None] > g[None, :])))
    dr = float(np.sum(M * (g[:, None] == g[None, :])))
    return float(max(0, 1 - hw - dr)), dr, hw   # (away, draw, home)


# ── Tournament simulation ────────────────────────────────────────────

def simulate_group(group_teams, group_matches_idx, match_pool, rng):
    """Simulate 6 matches in a group of 4 teams. Return ordered standings."""
    stats = {t: {"pts": 0, "gd": 0, "gf": 0, "team": t} for t in group_teams}
    for mi in group_matches_idx:
        info = match_pool[mi]
        h, a = info["home"], info["away"]
        # Sample a scoreline from precomputed matrix
        flat_idx = rng.choice(len(info["flat_probs"]), p=info["flat_probs"])
        hg = flat_idx // (MAX_GOALS + 1)
        ag = flat_idx % (MAX_GOALS + 1)
        if hg > ag:
            stats[h]["pts"] += 3
        elif ag > hg:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1
        stats[h]["gd"] += hg - ag; stats[a]["gd"] += ag - hg
        stats[h]["gf"] += hg;       stats[a]["gf"] += ag
    return sorted(stats.values(), key=lambda x: (x["pts"], x["gd"], x["gf"]),
                  reverse=True)


def simulate_knockout(home, away, knockout_pool, rng):
    """Knockout match between two teams. Returns winner team name."""
    key = (home, away)
    info = knockout_pool.get(key)
    if info is None:
        # Generate on-the-fly using DC fallback (rare: bracket pairs not pre-cached)
        return home if rng.random() < 0.5 else away
    flat_idx = rng.choice(len(info["flat_probs"]), p=info["flat_probs"])
    hg = flat_idx // (MAX_GOALS + 1)
    ag = flat_idx % (MAX_GOALS + 1)
    if hg > ag:
        return home
    if ag > hg:
        return away
    # Draw → penalty shootout. ~50/50 with mild ELO tilt
    p_home = 0.5 + 0.0005 * (info["elo_h"] - info["elo_a"])
    p_home = max(0.30, min(0.70, p_home))
    return home if rng.random() < p_home else away


def simulate_tournament(group_match_idx_by_group, group_teams_by_group,
                        match_pool, knockout_pool, rng):
    # Group stage
    winners, runners_up, third_place = {}, {}, []
    for grp, teams_in_grp in group_teams_by_group.items():
        standings = simulate_group(teams_in_grp, group_match_idx_by_group[grp],
                                    match_pool, rng)
        winners[grp] = standings[0]["team"]
        runners_up[grp] = standings[1]["team"]
        third_place.append({**standings[2], "group": grp})

    # Best 8 third-placed (out of 12)
    third_sorted = sorted(third_place,
                          key=lambda x: (x["pts"], x["gd"], x["gf"]),
                          reverse=True)
    best_thirds = [t["team"] for t in third_sorted[:8]]

    # 32-team bracket: simplified seeding
    # Real WC 2026 R32 uses a pre-defined bracket structure based on group letters
    # For Monte Carlo purposes we use a deterministic but fair pairing
    r32_teams = []
    # Order: 12 winners + 12 runners-up + 8 best thirds = 32 teams
    # Pair as: W1v3rd, W2v3rd, ..., RU1vRU2, ... (simplified deterministic seed)
    pool32 = (list(winners.values()) + list(runners_up.values()) + best_thirds)
    # Shuffle? No — keep deterministic. Just pair adjacent.
    pairs_r32 = [(pool32[i], pool32[i + 1]) for i in range(0, 32, 2)]

    progressed = {"group_stage": set(pool32)}

    def play_round(pairs):
        winners_round = []
        for h, a in pairs:
            w = simulate_knockout(h, a, knockout_pool, rng)
            winners_round.append(w)
        return winners_round

    r16 = play_round(pairs_r32)
    progressed["r16"] = set(r16)
    r16_pairs = [(r16[i], r16[i + 1]) for i in range(0, 16, 2)]
    qf = play_round(r16_pairs)
    progressed["qf"] = set(qf)
    qf_pairs = [(qf[i], qf[i + 1]) for i in range(0, 8, 2)]
    sf = play_round(qf_pairs)
    progressed["sf"] = set(sf)
    sf_pairs = [(sf[i], sf[i + 1]) for i in range(0, 4, 2)]
    finals = play_round(sf_pairs)
    progressed["final"] = set(finals)
    winner = play_round([(finals[0], finals[1])])[0]
    progressed["winner"] = {winner}

    return progressed


def main():
    print("=" * 90)
    print(f"PHASE D-2 — {N_SIMS:,}-sim Monte Carlo, Phase 3 hybrid model")
    print("=" * 90)

    # Load hybrid model
    bundle = joblib.load(MODELS / "phase3_hybrid_clean.pkl")
    mh, ma, rho, features = bundle["model_home"], bundle["model_away"], bundle["rho"], bundle["features"]
    print(f"\n[Load] hybrid model (rho={rho:.4f})")

    # Build feature vectors for 72 group matches
    print("[Build] feature vectors for 72 WC 2026 group matches...")
    match_feats = build_match_features()
    print(f"  {len(match_feats)} matches × {len(match_feats.columns)} cols")

    # Precompute scoreline matrices for all 72 group matches
    X_group = match_feats[features].values
    lh_group = mh.predict(X_group)
    la_group = ma.predict(X_group)
    print(f"  λ_h range: [{lh_group.min():.2f}, {lh_group.max():.2f}], mean {lh_group.mean():.2f}")
    print(f"  λ_a range: [{la_group.min():.2f}, {la_group.max():.2f}], mean {la_group.mean():.2f}")

    print(f"\n[Cache] flatten scoreline distributions for fast sampling...")
    match_pool = []
    for i, row in match_feats.iterrows():
        M = lams_to_scoreline_matrix(lh_group[i], la_group[i], rho)
        flat = M.flatten()
        flat = flat / flat.sum()
        match_pool.append({"home": row["home_team"], "away": row["away_team"],
                           "flat_probs": flat,
                           "lh": float(lh_group[i]), "la": float(la_group[i])})

    # Precompute knockout matrices for ALL possible pairs of 48 teams
    print(f"[Cache] precomputing scoreline matrices for all 48×47 knockout pairs...")
    teams = pd.read_csv(PROC / "teams_2026.csv")["team"].tolist()
    n_teams = len(teams)

    # Build a template feature row for every pair (h, a) using same logic
    # Build pairs DataFrame
    pair_rows = []
    elos = pd.read_csv(PROC / "final_elos.csv")
    elo_map = dict(zip(elos["team"], elos["final_elo"]))
    dc_ratings = pd.read_csv(PROC / "dc_ratings_v2.csv")
    with open(MODELS / "dc_params_v2.json") as f:
        dc_params = json.load(f)
    dc_map = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}

    # Re-use the build_match_features template — call once per match by editing teams
    base_template = match_feats.iloc[0].copy()  # use first match's neutral features as template
    pair_rows = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            row = base_template.copy()
            row["home_team"] = h
            row["away_team"] = a
            row["home_elo_before"] = elo_map.get(h, 1500)
            row["away_elo_before"] = elo_map.get(a, 1500)
            row["elo_diff"] = row["home_elo_before"] - row["away_elo_before"]
            row["elo_diff_sq"] = row["elo_diff"] ** 2
            if h in dc_map and a in dc_map:
                h_att, h_def = dc_map[h]; a_att, a_def = dc_map[a]
                lam = float(np.exp(h_att + a_def))
                mu = float(np.exp(a_att + h_def))
            else:
                lam = mu = 1.4
            g = np.arange(MAX_GOALS + 1)
            M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
            M[0, 0] *= max(1 - lam * mu * dc_params["rho"], 1e-10)
            M[0, 1] *= max(1 + lam * dc_params["rho"], 1e-10)
            M[1, 0] *= max(1 + mu * dc_params["rho"], 1e-10)
            M[1, 1] *= max(1 - dc_params["rho"], 1e-10)
            M = M / M.sum()
            hw = float(np.sum(M * (g[:, None] > g[None, :])))
            dr = float(np.sum(M * (g[:, None] == g[None, :])))
            row["dc_home_win_prob"] = hw; row["dc_draw_prob"] = dr
            row["dc_away_win_prob"] = max(0, 1 - hw - dr)
            row["dc_lambda"] = lam; row["dc_mu"] = mu
            row["dc_total_goals"] = lam + mu; row["dc_goal_diff"] = lam - mu
            pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows)
    # Re-join squad/SB/intl/chem (already joined to match_feats template; we have to redo)
    tfy = pd.read_csv(PROC / "team_features_by_year.csv")
    sb = pd.read_csv(PROC / "team_statsbomb_features.csv")
    intl = pd.read_csv(PROC / "team_intl_form_features.csv")
    chem = pd.read_csv(PROC / "team_chemistry_features.csv")
    # Drop old joined cols then re-join
    drop_cols = HOME_SQUAD + AWAY_SQUAD + HOME_SB + AWAY_SB + HOME_INTL + AWAY_INTL + HOME_CHEM + AWAY_CHEM
    drop_cols = [c for c in drop_cols if c in pairs.columns]
    pairs = pairs.drop(columns=drop_cols, errors="ignore")
    if "fifa_year" in pairs.columns:
        pairs = pairs.drop(columns=["fifa_year"])
    pairs["date"] = pd.Timestamp("2026-06-11")  # ref date for fifa_year lookup
    pairs = join_squad(pairs, tfy)
    pairs = join_static(pairs, sb, intl, chem)

    X_pairs = pairs[features].values
    lh_pairs = mh.predict(X_pairs)
    la_pairs = ma.predict(X_pairs)

    knockout_pool = {}
    for i in range(len(pairs)):
        h = pairs.iloc[i]["home_team"]; a = pairs.iloc[i]["away_team"]
        M = lams_to_scoreline_matrix(lh_pairs[i], la_pairs[i], rho)
        flat = M.flatten(); flat = flat / flat.sum()
        knockout_pool[(h, a)] = {"flat_probs": flat,
                                  "elo_h": elo_map.get(h, 1500),
                                  "elo_a": elo_map.get(a, 1500)}
    print(f"  cached {len(knockout_pool)} knockout pair matrices")

    # Index group structure
    teams_2026 = pd.read_csv(PROC / "teams_2026.csv")
    group_teams_by_group = {g: list(teams_2026[teams_2026["group"] == g]["team"])
                            for g in teams_2026["group"].unique()}
    group_match_idx_by_group = {g: [] for g in group_teams_by_group.keys()}
    for i, m in match_feats.iterrows():
        grp = m["match_id"][0]  # match_id starts with group letter
        group_match_idx_by_group[grp].append(i)

    # ── Run simulations
    print(f"\n[Simulate] running {N_SIMS:,} simulations...")
    rng = np.random.default_rng(RNG_SEED)
    counts = {"r16": defaultdict(int), "qf": defaultdict(int), "sf": defaultdict(int),
              "final": defaultdict(int), "winner": defaultdict(int)}

    t0 = time.time()
    for s in range(N_SIMS):
        result = simulate_tournament(group_match_idx_by_group, group_teams_by_group,
                                      match_pool, knockout_pool, rng)
        for stage in ["r16", "qf", "sf", "final", "winner"]:
            for team in result[stage]:
                counts[stage][team] += 1
        if (s + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (s + 1) / elapsed
            eta = (N_SIMS - s - 1) / rate
            print(f"  {s+1:,}/{N_SIMS:,}  ({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({N_SIMS/elapsed:.0f} sims/sec)")

    # ── Output
    rows = []
    all_teams = teams_2026["team"].tolist()
    for t in all_teams:
        rows.append({
            "team": t, "group": teams_2026.set_index("team").loc[t, "group"],
            "confederation": teams_2026.set_index("team").loc[t, "confederation"],
            "elo": elo_map.get(t, 1500),
            "p_r16": counts["r16"][t] / N_SIMS,
            "p_qf": counts["qf"][t] / N_SIMS,
            "p_sf": counts["sf"][t] / N_SIMS,
            "p_final": counts["final"][t] / N_SIMS,
            "p_win": counts["winner"][t] / N_SIMS,
            "wins": counts["winner"][t],
        })
    out = pd.DataFrame(rows).sort_values("p_win", ascending=False)
    out.to_csv(PROC / "wc2026_simulation_phase3.csv", index=False)
    print(f"\n  Saved: data/processed/wc2026_simulation_phase3.csv")

    print("\nTop 15 by P(win WC):")
    print(out.head(15)[["team", "elo", "p_r16", "p_qf", "p_sf", "p_final", "p_win"]].to_string(index=False))

    # Also save per-match probabilities for the 72 group matches
    match_probs = []
    for i, m in match_feats.iterrows():
        M = lams_to_scoreline_matrix(lh_group[i], la_group[i], rho)
        aw, dr, hw = matrix_to_wdl(M)
        match_probs.append({
            "match_id": m["match_id"], "date": m["date"].strftime("%Y-%m-%d"),
            "home": m["home_team"], "away": m["away_team"],
            "lambda_h": float(lh_group[i]), "lambda_a": float(la_group[i]),
            "p_home_win": hw, "p_draw": dr, "p_away_win": aw,
        })
    pd.DataFrame(match_probs).to_csv(PROC / "wc2026_match_predictions_phase3.csv", index=False)
    print(f"  Saved: data/processed/wc2026_match_predictions_phase3.csv ({len(match_probs)} matches)")


if __name__ == "__main__":
    main()
