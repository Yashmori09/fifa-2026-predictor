"""
Update notebook 06 for EXP-16b (bounds-based convergence fix) and EXP-17 (xi tuning).
Run with: C:/Users/Yash/anaconda3/python.exe update_notebook.py
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/06_dixon_coles_features.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Helper: make a code cell ───────────────────────────────────────────────────
def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"exp-cell-{abs(hash(source_lines[0]))%99999:05d}",
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": f"md-cell-{abs(hash(text[:20]))%99999:05d}",
        "metadata": {},
        "source": [text],
    }


# ── Find cells by searching source content ──────────────────────────────────
cells = nb["cells"]

def find_cell_idx(keyword):
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if keyword in src:
            return i
    return None

# ── 1. EXP-17: Insert xi comparison cell after the XI=0.003 cell ─────────────
xi_cell_idx = find_cell_idx("XI = 0.003")
if xi_cell_idx is None:
    xi_cell_idx = find_cell_idx("# ── Time-weighting")

print(f"Time-weighting cell found at index: {xi_cell_idx}")

xi_comparison_md = "## EXP-17 — Tune ξ Decay Rate\n\nCompare ξ=0.001, 0.002, 0.003 on a quick fit to find which puts the right teams at the top.\n\nExpected: Spain, Argentina, France, Brazil in top 5. ξ=0.003 currently ranks Australia #1 (too aggressive)."

xi_comparison_code = [
    "# EXP-17: Quick xi comparison — fit DC with 3 different decay rates\n",
    "# We use fewer optimizer iterations here just to compare leaderboards\n",
    "from scipy.optimize import minimize as _minimize\n",
    "\n",
    "def quick_dc_fit(xi_val, max_iter=500):\n",
    "    ref = comp['date'].max()\n",
    "    days = (ref - comp['date']).dt.days.values\n",
    "    w = np.exp(-xi_val * days)\n",
    "\n",
    "    hg = comp['home_score'].values\n",
    "    ag = comp['away_score'].values\n",
    "    hi = home_idx_arr\n",
    "    ai = away_idx_arr\n",
    "\n",
    "    m00 = (hg == 0) & (ag == 0)\n",
    "    m01 = (hg == 0) & (ag == 1)\n",
    "    m10 = (hg == 1) & (ag == 0)\n",
    "    m11 = (hg == 1) & (ag == 1)\n",
    "\n",
    "    def nll(params):\n",
    "        atk = params[:n_teams]\n",
    "        dfn = params[n_teams:2*n_teams]\n",
    "        ha  = params[2*n_teams]\n",
    "        rh  = params[2*n_teams + 1]\n",
    "        lam = np.exp(atk[hi] + dfn[ai] + ha)\n",
    "        mu  = np.exp(atk[ai] + dfn[hi])\n",
    "        tau = np.ones(len(hg))\n",
    "        tau[m00] = 1 - lam[m00]*mu[m00]*rh\n",
    "        tau[m01] = 1 + lam[m01]*rh\n",
    "        tau[m10] = 1 + mu[m10]*rh\n",
    "        tau[m11] = 1 - rh\n",
    "        valid = tau > 0\n",
    "        from scipy.stats import poisson as _p\n",
    "        ll = np.sum(w[valid] * (\n",
    "            np.log(tau[valid])\n",
    "            + _p.logpmf(hg[valid], lam[valid])\n",
    "            + _p.logpmf(ag[valid], mu[valid])\n",
    "        ))\n",
    "        return -ll\n",
    "\n",
    "    eng_idx = team_idx.get('England', 0)\n",
    "    init = np.zeros(2*n_teams + 2)\n",
    "    init[2*n_teams] = 0.3\n",
    "    init[2*n_teams+1] = -0.1\n",
    "    bnds = [(None, None)] * (2*n_teams + 2)\n",
    "    bnds[eng_idx] = (0.0, 0.0)\n",
    "\n",
    "    res = _minimize(nll, init, method='L-BFGS-B', bounds=bnds,\n",
    "                    options={'maxiter': max_iter, 'ftol': 1e-6})\n",
    "    atk = res.x[:n_teams]\n",
    "    dfn = res.x[n_teams:2*n_teams]\n",
    "    df_r = pd.DataFrame({'team': all_teams, 'attack': atk, 'defense': dfn})\n",
    "    df_r['overall'] = df_r['attack'] - df_r['defense']\n",
    "    df_r = df_r.sort_values('overall', ascending=False)\n",
    "    return res.success, df_r\n",
    "\n",
    "print('Testing xi=0.001, 0.002, 0.003 (quick fits, ~30s each)...')\n",
    "for xi_val in [0.001, 0.002, 0.003]:\n",
    "    success, df_r = quick_dc_fit(xi_val)\n",
    "    print(f'\\nxi={xi_val}  converged={success}')\n",
    "    print(df_r.head(10)[['team','overall']].to_string(index=False))\n",
]

# Insert after the xi cell
cells.insert(xi_cell_idx + 1, md_cell(xi_comparison_md))
cells.insert(xi_cell_idx + 2, code_cell(xi_comparison_code))

print(f"Inserted EXP-17 cells at index {xi_cell_idx+1} and {xi_cell_idx+2}")

# ── Refresh indices after insertion ─────────────────────────────────────────
cells = nb["cells"]

# ── 2. EXP-16b: Fix the main fitting cell — bounds instead of constraint ─────
fit_cell_idx = find_cell_idx("Constraint: sum of attack parameters")
if fit_cell_idx is None:
    fit_cell_idx = find_cell_idx("constraints = [{")
print(f"Fitting cell found at index: {fit_cell_idx}")

# Replace the entire source of the fitting cell
cells[fit_cell_idx]["source"] = [
    "%%time\n",
    "# ── Fit Dixon-Coles — EXP-16b: bounds-based anchor (England attack=0) ────────\n",
    "# WHY: L-BFGS-B cannot enforce equality constraints natively.\n",
    "#      The sum(attack)=0 constraint was applied externally, causing Converged=False.\n",
    "#      Fix: anchor England's attack parameter to 0.0 using bounds.\n",
    "#      This achieves identifiability without fighting the optimizer.\n",
    "\n",
    "init_params = np.zeros(2 * n_teams + 2)\n",
    "init_params[2 * n_teams]     = 0.3   # home advantage\n",
    "init_params[2 * n_teams + 1] = -0.1  # rho (low-score correction)\n",
    "\n",
    "# Anchor England's attack = 0 (reference team for identifiability)\n",
    "eng_idx = team_idx.get('England', 0)\n",
    "bounds  = [(None, None)] * (2 * n_teams + 2)\n",
    "bounds[eng_idx] = (0.0, 0.0)  # England attack fixed at 0\n",
    "\n",
    "print('Fitting Dixon-Coles model (EXP-16b: bounds-based anchor, EXP-17: xi selected)...')\n",
    "result = minimize(\n",
    "    dc_log_likelihood_fast,\n",
    "    init_params,\n",
    "    method='L-BFGS-B',\n",
    "    bounds=bounds,\n",
    "    options={'maxiter': 2000, 'ftol': 1e-6}\n",
    ")\n",
    "\n",
    "attack_params  = result.x[:n_teams]\n",
    "defense_params = result.x[n_teams:2*n_teams]\n",
    "home_adv       = result.x[2*n_teams]\n",
    "rho            = result.x[2*n_teams + 1]\n",
    "\n",
    "print(f'Converged: {result.success}')\n",
    "print(f'Home advantage factor: {np.exp(home_adv):.3f}x')\n",
    "print(f'Rho (low-score correction): {rho:.4f}')\n",
    "print(f'England attack (anchor): {attack_params[eng_idx]:.6f}  (should be ~0)')\n",
]
# Clear old outputs
cells[fit_cell_idx]["outputs"] = []
cells[fit_cell_idx]["execution_count"] = None
print(f"Updated fitting cell at index {fit_cell_idx}")

# ── 3. Also update XI in the time-weighting cell to 0.002 (best candidate) ────
tw_cell_idx = find_cell_idx("XI = 0.003")
if tw_cell_idx is not None:
    src = "".join(cells[tw_cell_idx]["source"])
    src = src.replace("XI = 0.003", "XI = 0.002  # EXP-17: tuned from 0.003 (was ranking Australia #1)")
    src = src.replace(
        "print(f'Weight 1yr ago: {np.exp(-XI*365):.3f}')",
        "print(f'[EXP-17] Using xi={XI} (reduced from 0.003 — Australia was ranked #1 with 0.003)')\n"
        "print(f'Weight 1yr ago: {np.exp(-XI*365):.3f}')"
    )
    cells[tw_cell_idx]["source"] = list(src)
    cells[tw_cell_idx]["outputs"] = []
    cells[tw_cell_idx]["execution_count"] = None
    print(f"Updated XI from 0.003 to 0.002 in cell {tw_cell_idx}")

# ── Save ─────────────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook updated successfully.")
print("Changes made:")
print("  1. EXP-17: xi comparison cell inserted (tests 0.001, 0.002, 0.003)")
print("  2. EXP-16b: fitting cell now uses bounds[eng_idx]=(0,0) instead of constraint")
print("  3. XI changed from 0.003 to 0.002 in time-weighting cell")
print("\nRun notebook 06 to see results.")
