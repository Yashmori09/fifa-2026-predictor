"""
Generate pairwise probabilities + bracket structure for the /what-if page.

For each pair of the 32 R32 teams, compute P(A beats B) in a neutral knockout
match (draws split 50/50 to penalties). Uses the Phase 3 hybrid model.

Output: frontend/public/whatif-data.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from phase3d_simulate import (
    build_match_features, join_squad, join_static,
    HOME_SQUAD, AWAY_SQUAD, HOME_SB, AWAY_SB, HOME_INTL, AWAY_INTL, HOME_CHEM, AWAY_CHEM,
)

OUT_JSON = ROOT / "frontend/public/whatif-data.json"
MAX_GOALS = 10


def compute_win_prob(lh: float, la: float, rho: float) -> float:
    """P(home wins) in a knockout — includes half of the draw prob (penalties)."""
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= max(1 - lh * la * rho, 1e-10)
    M[0, 1] *= max(1 + lh * rho, 1e-10)
    M[1, 0] *= max(1 + la * rho, 1e-10)
    M[1, 1] *= max(1 - rho, 1e-10)
    M = M / M.sum()
    hw = float(np.sum(M * (g[:, None] > g[None, :])))
    dr = float(np.sum(M * (g[:, None] == g[None, :])))
    return hw + dr * 0.5


def main():
    print("Loading model + refs...")
    bundle = joblib.load(ROOT / "models/phase3_hybrid_clean.pkl")
    mh, ma, rho = bundle["model_home"], bundle["model_away"], bundle["rho"]
    features = bundle["features"]

    elos = pd.read_csv(ROOT / "data/processed/final_elos.csv")
    elo_map = dict(zip(elos["team"], elos["final_elo"]))
    dc_ratings = pd.read_csv(ROOT / "data/processed/dc_ratings_v2.csv")
    with open(ROOT / "models/dc_params_v2.json") as f:
        dc_params = json.load(f)
    dc_map = {r["team"]: (r["attack"], r["defense"]) for _, r in dc_ratings.iterrows()}
    tfy = pd.read_csv(ROOT / "data/processed/team_features_by_year.csv")
    sb = pd.read_csv(ROOT / "data/processed/team_statsbomb_features.csv")
    intl = pd.read_csv(ROOT / "data/processed/team_intl_form_features.csv")
    chem = pd.read_csv(ROOT / "data/processed/team_chemistry_features.csv")

    mf = build_match_features()
    base_row = mf.iloc[0]

    # Load the shipped deterministic bracket to get R32 pairings
    with open(ROOT / "frontend/src/lib/deterministic-data.ts") as f:
        s = f.read()
    det = json.loads(s[s.index("{"):s.rindex("}") + 1])

    r32_teams = []
    for m in det["knockout"]["r32"]:
        r32_teams.append(m["team1"])
        r32_teams.append(m["team2"])

    print(f"Computing pairwise probs for {len(r32_teams)} R32 teams...")

    # Build all pair rows for prediction
    pair_rows = []
    for h in r32_teams:
        for a in r32_teams:
            if h == a:
                continue
            row = base_row.copy()
            row["home_team"] = h; row["away_team"] = a
            row["home_elo_before"] = elo_map.get(h, 1500)
            row["away_elo_before"] = elo_map.get(a, 1500)
            row["elo_diff"] = row["home_elo_before"] - row["away_elo_before"]
            row["elo_diff_sq"] = row["elo_diff"] ** 2
            if h in dc_map and a in dc_map:
                h_att, h_def = dc_map[h]; a_att, a_def = dc_map[a]
                lam = float(np.exp(h_att + a_def)); mu = float(np.exp(a_att + h_def))
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

    df = pd.DataFrame(pair_rows)
    drop_cols = HOME_SQUAD + AWAY_SQUAD + HOME_SB + AWAY_SB + HOME_INTL + AWAY_INTL + HOME_CHEM + AWAY_CHEM
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if "fifa_year" in df.columns:
        df = df.drop(columns=["fifa_year"])
    df["date"] = pd.Timestamp("2026-06-11")
    df = join_squad(df, tfy)
    df = join_static(df, sb, intl, chem)

    X = df[features].values
    lh_arr = mh.predict(X)
    la_arr = ma.predict(X)

    pair_win = {}
    for i in range(len(df)):
        h = df.iloc[i]["home_team"]
        a = df.iloc[i]["away_team"]
        p = compute_win_prob(lh_arr[i], la_arr[i], rho)
        pair_win[f"{h}||{a}"] = round(p, 4)

    # Symmetrize: neutral-venue P(A>B) = 0.5 * (P(A@home,B) + (1 - P(B@home,A)))
    # This removes the small home/away asymmetry from features.
    sym_pair_win = {}
    for h in r32_teams:
        for a in r32_teams:
            if h == a:
                continue
            p_ha = pair_win.get(f"{h}||{a}")
            p_ah = pair_win.get(f"{a}||{h}")
            if p_ha is None or p_ah is None:
                continue
            sym = 0.5 * (p_ha + (1 - p_ah))
            sym_pair_win[f"{h}||{a}"] = round(sym, 4)

    # Bracket structure (from deterministic-data.ts + backend pairings)
    sys.path.insert(0, str(ROOT / "backend"))
    from app.core.simulator import R16_PAIRINGS, QF_PAIRINGS, SF_PAIRINGS

    # Load actual results from live API for the ACTUAL brackets
    import urllib.request
    print("Fetching actual tournament results...")
    live = json.loads(urllib.request.urlopen(
        "https://fifa-2026-predictor.vercel.app/api/live-matches"
    ).read())

    actual_by_stage = {"r32": [], "r16": [], "qf": [], "sf": [], "final": []}
    for m in live["matches"]:
        stage = m.get("stage")
        if stage not in actual_by_stage:
            continue
        if m["status"] != "FINISHED":
            continue
        act = m.get("actual") or {}
        if not act:
            continue
        winner = m["home"]["name"] if act["outcome"] == "home" else \
                 m["away"]["name"] if act["outcome"] == "away" else None
        actual_by_stage[stage].append({
            "team1": m["home"]["name"],
            "team2": m["away"]["name"],
            "team1_goals": act["score"]["home"],
            "team2_goals": act["score"]["away"],
            "winner": winner,
        })

    # Build the bracket TREE — each match has a stable id + parent match ids.
    # We use the ACTUAL tournament bracket (from live API) for the /what-if page
    # so users can flip real matchups.
    #
    # For a proper tree, we need to know for each R16 match: which R32 matches
    # fed its team1 and team2. We infer this by matching: R16.team1 = winner of
    # the R32 match where team1 is one of {team1, team2} AND won.
    def find_parent(team, parent_stage_matches):
        """Find the match at parent_stage where `team` was the winner."""
        for i, m in enumerate(parent_stage_matches):
            if m.get("winner") == team:
                return i
        return None

    tree = {"r32": [], "r16": [], "qf": [], "sf": [], "final": []}
    for i, m in enumerate(actual_by_stage["r32"]):
        tree["r32"].append({
            "id": f"r32_{i}",
            "stage": "r32",
            "team1": m["team1"], "team2": m["team2"],
            "team1_goals": m["team1_goals"], "team2_goals": m["team2_goals"],
            "winner": m["winner"],
            "parent1": None, "parent2": None,
        })
    for i, m in enumerate(actual_by_stage["r16"]):
        p1 = find_parent(m["team1"], actual_by_stage["r32"])
        p2 = find_parent(m["team2"], actual_by_stage["r32"])
        tree["r16"].append({
            "id": f"r16_{i}",
            "stage": "r16",
            "team1": m["team1"], "team2": m["team2"],
            "team1_goals": m["team1_goals"], "team2_goals": m["team2_goals"],
            "winner": m["winner"],
            "parent1": f"r32_{p1}" if p1 is not None else None,
            "parent2": f"r32_{p2}" if p2 is not None else None,
        })
    for i, m in enumerate(actual_by_stage["qf"]):
        p1 = find_parent(m["team1"], actual_by_stage["r16"])
        p2 = find_parent(m["team2"], actual_by_stage["r16"])
        tree["qf"].append({
            "id": f"qf_{i}",
            "stage": "qf",
            "team1": m["team1"], "team2": m["team2"],
            "team1_goals": m["team1_goals"], "team2_goals": m["team2_goals"],
            "winner": m["winner"],
            "parent1": f"r16_{p1}" if p1 is not None else None,
            "parent2": f"r16_{p2}" if p2 is not None else None,
        })
    for i, m in enumerate(actual_by_stage["sf"]):
        p1 = find_parent(m["team1"], actual_by_stage["qf"])
        p2 = find_parent(m["team2"], actual_by_stage["qf"])
        tree["sf"].append({
            "id": f"sf_{i}",
            "stage": "sf",
            "team1": m["team1"], "team2": m["team2"],
            "team1_goals": m["team1_goals"], "team2_goals": m["team2_goals"],
            "winner": m["winner"],
            "parent1": f"qf_{p1}" if p1 is not None else None,
            "parent2": f"qf_{p2}" if p2 is not None else None,
        })
    for i, m in enumerate(actual_by_stage["final"]):
        p1 = find_parent(m["team1"], actual_by_stage["sf"])
        p2 = find_parent(m["team2"], actual_by_stage["sf"])
        tree["final"].append({
            "id": f"final_{i}",
            "stage": "final",
            "team1": m["team1"], "team2": m["team2"],
            "team1_goals": m["team1_goals"], "team2_goals": m["team2_goals"],
            "winner": m["winner"],
            "parent1": f"sf_{p1}" if p1 is not None else None,
            "parent2": f"sf_{p2}" if p2 is not None else None,
        })

    payload = {
        "generated_at": live["last_updated"],
        "tree": tree,
        "pair_win": sym_pair_win,
        "r32_teams": r32_teams,
        "champion": tree["final"][0]["winner"] if tree["final"] else None,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUT_JSON}")
    print(f"  R32 teams: {len(r32_teams)}")
    print(f"  Pair probs: {len(sym_pair_win)}")

    # Sanity check
    print(f"\nSample: P(Spain beats Argentina) = {sym_pair_win.get('Spain||Argentina')}")
    print(f"Sample: P(Argentina beats Spain) = {sym_pair_win.get('Argentina||Spain')}")


if __name__ == "__main__":
    main()
