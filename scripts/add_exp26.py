"""Add EXP-26 cells to notebook 06."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/06_dixon_coles_features.ipynb")
nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# Find fitting cell
fit_idx = next(i for i, c in enumerate(cells) if "bounds[eng_idx]" in "".join(c.get("source", [])))
print(f"Fitting cell at index: {fit_idx}")

md_26 = {
    "cell_type": "markdown",
    "id": "exp26-md",
    "metadata": {},
    "source": [
        "## EXP-26 — Refit DC on Top-Tier Matches Only\n\n",
        "Qualifier/regional matches inflate weak-confederation teams (Japan #1, Australia #2).\n",
        "Fix: fit DC only on World Cup + major continental tournaments.\n",
        "6,353 matches across 201 teams — cleaner signal, no qualifier noise.\n",
        "Teams not in top-tier get their ratings from the full fit as fallback.",
    ],
}

code_26_src = '''\
%%time
# EXP-26: Refit DC on top-tier matches only (WC + major continentals)
TOP_TIER = [
    "FIFA World Cup", "World Cup",
    "Copa America", "Copa Am\u00e9rica",
    "African Cup of Nations",
    "UEFA Euro", "European Championship", "UEFA Nations League",
    "AFC Asian Cup", "Asian Cup",
    "Gold Cup", "CONCACAF Championship", "CONCACAF Nations League",
    "Confederations Cup",
]

comp_top = matches[
    matches["tournament"].isin(TOP_TIER) &
    matches["home_score"].notna() &
    matches["away_score"].notna()
].copy()
comp_top["home_score"] = comp_top["home_score"].astype(int)
comp_top["away_score"] = comp_top["away_score"].astype(int)
comp_top["days_ago"]   = (comp_top["date"].max() - comp_top["date"]).dt.days
comp_top["weight"]     = np.exp(-XI * comp_top["days_ago"])

print(f"Top-tier matches: {len(comp_top):,}  (was 38,617 with all competitive)")
n_top_teams = len(set(comp_top["home_team"]) | set(comp_top["away_team"]))
print(f"Unique teams: {n_top_teams}")

# Build index arrays for top-tier subset
all_teams_top = sorted(set(comp_top["home_team"]) | set(comp_top["away_team"]))
team_idx_top  = {t: i for i, t in enumerate(all_teams_top)}
n_teams_top   = len(all_teams_top)

ht_top = np.array([team_idx_top[t] for t in comp_top["home_team"].values])
at_top = np.array([team_idx_top[t] for t in comp_top["away_team"].values])
hg_top = comp_top["home_score"].values
ag_top = comp_top["away_score"].values
w_top  = comp_top["weight"].values

m00t = (hg_top == 0) & (ag_top == 0)
m01t = (hg_top == 0) & (ag_top == 1)
m10t = (hg_top == 1) & (ag_top == 0)
m11t = (hg_top == 1) & (ag_top == 1)

def dc_nll_top(params):
    atk = params[:n_teams_top]
    dfn = params[n_teams_top:2*n_teams_top]
    ha  = params[2*n_teams_top]
    rh  = params[2*n_teams_top + 1]
    lam = np.exp(atk[ht_top] + dfn[at_top] + ha)
    mu  = np.exp(atk[at_top] + dfn[ht_top])
    tau = np.ones(len(hg_top))
    tau[m00t] = 1 - lam[m00t]*mu[m00t]*rh
    tau[m01t] = 1 + lam[m01t]*rh
    tau[m10t] = 1 + mu[m10t]*rh
    tau[m11t] = 1 - rh
    valid = tau > 0
    ll = np.sum(w_top[valid] * (
        np.log(tau[valid])
        + poisson.logpmf(hg_top[valid], lam[valid])
        + poisson.logpmf(ag_top[valid], mu[valid])
    ))
    return -ll

eng_idx_top = team_idx_top.get("England", 0)
init_top    = np.zeros(2*n_teams_top + 2)
init_top[2*n_teams_top]     = 0.3
init_top[2*n_teams_top + 1] = -0.1
bnds_top    = [(None, None)] * (2*n_teams_top + 2)
bnds_top[eng_idx_top] = (0.0, 0.0)

print("Fitting DC on top-tier matches...")
res_top = minimize(dc_nll_top, init_top, method="L-BFGS-B", bounds=bnds_top,
                   options={"maxiter": 2000, "ftol": 1e-6})

atk_top = res_top.x[:n_teams_top]
dfn_top = res_top.x[n_teams_top:2*n_teams_top]
ha_top  = res_top.x[2*n_teams_top]
rho_top = res_top.x[2*n_teams_top + 1]

print(f"Converged: {res_top.success}")
print(f"Home advantage: {np.exp(ha_top):.3f}x")
print(f"Rho: {rho_top:.4f}")

# Leaderboard
df_top = pd.DataFrame({"team": all_teams_top, "attack": atk_top, "defense": dfn_top})
df_top["overall"] = df_top["attack"] - df_top["defense"]
df_top = df_top.sort_values("overall", ascending=False)
print("\\nTop 20 (EXP-26 top-tier DC):")
print(df_top.head(20)[["team", "overall"]].to_string(index=False))

# Merge: use top-tier params where available, full-fit as fallback for unknowns
attack_params_26  = attack_params.copy()
defense_params_26 = defense_params.copy()
for team, idx_full in team_idx.items():
    if team in team_idx_top:
        attack_params_26[idx_full]  = atk_top[team_idx_top[team]]
        defense_params_26[idx_full] = dfn_top[team_idx_top[team]]

# Overwrite global params used by all downstream cells
attack_params  = attack_params_26
defense_params = defense_params_26
home_adv       = ha_top
rho            = rho_top

print(f"\\nParams updated. {len(all_teams_top)}/{len(all_teams)} teams have top-tier ratings.")
print("All downstream cells (leaderboard, DC features, ensemble) now use EXP-26 params.")
'''

code_26 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "exp26-code",
    "metadata": {},
    "outputs": [],
    "source": code_26_src.splitlines(keepends=True),
}

cells.insert(fit_idx + 1, md_26)
cells.insert(fit_idx + 2, code_26)

nb["cells"] = cells
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"Done. EXP-26 inserted at cells {fit_idx+1} and {fit_idx+2}. Total cells: {len(cells)}")
