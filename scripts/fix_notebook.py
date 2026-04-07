"""
Fix the messed up notebook — cell 4 source was exploded into individual chars.
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/06_dixon_coles_features.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Fix any cell whose source is a list of single characters ─────────────────
for i, c in enumerate(cells):
    src = c.get("source", [])
    if isinstance(src, list) and len(src) > 5:
        single_char_count = sum(1 for s in src if len(s) == 1)
        if single_char_count > len(src) * 0.5:
            full_str = "".join(src)
            lines = []
            for line in full_str.split("\n"):
                lines.append(line + "\n")
            if lines:
                lines[-1] = lines[-1].rstrip("\n")
                if lines[-1] == "":
                    lines.pop()
            c["source"] = lines

# ── Rewrite cell 4 (time-weighting) with clean ASCII-safe content ─────────────
tw_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "XI = 0.002" in src or "XI = 0.003" in src:
        tw_idx = i
        break

cells[tw_idx]["source"] = [
    "# -- Time-weighting (EXP-17: xi=0.002 instead of 0.003) --\n",
    "# xi=0.003 ranked Australia #1 (too aggressive, recent weak-opponent wins inflated)\n",
    "# xi=0.002 gives better balance: 1yr ago = 48%, 3yr ago = 11%, 5yr ago = 2.6%\n",
    "XI = 0.002\n",
    "reference_date = comp['date'].max()\n",
    "comp['days_ago'] = (reference_date - comp['date']).dt.days\n",
    "comp['weight']   = np.exp(-XI * comp['days_ago'])\n",
    "\n",
    "print(f'Reference date: {reference_date.date()}')\n",
    "print(f'[EXP-17] xi={XI}')\n",
    "print(f'Weight range: {comp[\"weight\"].min():.4f} -> {comp[\"weight\"].max():.4f}')\n",
    "print(f'Weight 1yr ago: {np.exp(-XI*365):.3f}')\n",
    "print(f'Weight 3yr ago: {np.exp(-XI*365*3):.3f}')\n",
    "print(f'Weight 5yr ago: {np.exp(-XI*365*5):.3f}')\n",
]
cells[tw_idx]["outputs"] = []
cells[tw_idx]["execution_count"] = None

# ── Rewrite fitting cell to use bounds (EXP-16b) ─────────────────────────────
fit_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "Constraint: sum of attack" in src or "constraints = [{" in src or "bounds[eng_idx]" in src:
        fit_idx = i
        break

cells[fit_idx]["source"] = [
    "%%time\n",
    "# -- Fit Dixon-Coles -- EXP-16b: bounds anchor instead of equality constraint --\n",
    "# L-BFGS-B handles bounds natively but NOT equality constraints.\n",
    "# Fix: set bounds[eng_idx] = (0.0, 0.0) to anchor England attack=0.\n",
    "\n",
    "init_params = np.zeros(2 * n_teams + 2)\n",
    "init_params[2 * n_teams]     = 0.3\n",
    "init_params[2 * n_teams + 1] = -0.1\n",
    "\n",
    "eng_idx = team_idx.get('England', 0)\n",
    "bounds  = [(None, None)] * (2 * n_teams + 2)\n",
    "bounds[eng_idx] = (0.0, 0.0)  # anchor England attack = 0 (identifiability)\n",
    "\n",
    "print('Fitting Dixon-Coles model (EXP-16b + EXP-17)...')\n",
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
    "print(f'England attack anchor check: {attack_params[eng_idx]:.6f}  (should be ~0.0)')\n",
]
cells[fit_idx]["outputs"] = []
cells[fit_idx]["execution_count"] = None

# ── SAVE FIRST, then print summary ───────────────────────────────────────────
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook saved. Total cells: {len(cells)}")
print(f"Time-weighting cell: {tw_idx}  (XI=0.002, no unicode chars)")
print(f"Fitting cell: {fit_idx}  (bounds-based, EXP-16b)")

# Verify
c4_src = "".join(cells[tw_idx]["source"])
c_fit_src = "".join(cells[fit_idx]["source"])
print(f"Cell {tw_idx} source items: {len(cells[tw_idx]['source'])}  (should be ~14, not 744)")
print(f"Cell {fit_idx} has bounds: {'bounds[eng_idx]' in c_fit_src}")
print(f"Cell {fit_idx} has old constraint: {'constraints = [{' in c_fit_src}")
print("Done.")
