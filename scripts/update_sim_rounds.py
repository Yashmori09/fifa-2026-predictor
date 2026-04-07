import json

with open('D:/project/fifa-2026-predictor/notebooks/07_tournament_simulation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Update simulate_tournament (cell 15)
nb['cells'][15]['source'] = [
"""def simulate_tournament(verbose=False):
    winners, runners_up, best_thirds, third_standings = simulate_all_groups()

    # Track furthest round each team reaches
    all_48 = [t for grp in groups_2026.values() for t in grp]
    round_reached = {t: "Group" for t in all_48}

    r32_qualified = list(winners.values()) + list(runners_up.values()) + best_thirds
    for t in r32_qualified:
        round_reached[t] = "R32"

    if verbose:
        print("=== GROUP STAGE ===")
        for g in sorted(winners):
            w = winners[g]; r = runners_up[g]
            print(f"  Group {g}: 1st={w}  2nd={r}")

    r32_matchups = build_r32_bracket(winners, runners_up, best_thirds)
    bracket = [t for match in r32_matchups for t in match]
    round_names = ["R16", "QF", "SF", "Final"]
    round_num = 0
    teams = bracket

    while len(teams) > 1:
        next_round = []
        for i in range(0, len(teams), 2):
            t1, t2 = teams[i], teams[i+1]
            winner = simulate_ko_match(t1, t2)
            if round_num + 1 < len(round_names):
                round_reached[winner] = round_names[round_num + 1]
            else:
                round_reached[winner] = "Champion"
            if verbose:
                print(f"  {t1:20s} vs {t2:20s}  -> {winner}")
            next_round.append(winner)
        teams = next_round
        round_num += 1

    champion = teams[0]
    round_reached[champion] = "Champion"
    if verbose:
        print(f"CHAMPION: {champion}")
    return champion, round_reached

print("simulate_tournament updated - now returns (champion, round_reached)")
"""
]
nb['cells'][15]['outputs'] = []
nb['cells'][15]['execution_count'] = None

# Update Monte Carlo cell (18)
nb['cells'][18]['source'] = [
"""%%time
N_SIMS = 10_000
champion_counts = {}
round_counts = {}  # team -> {round: count}

for _ in tqdm(range(N_SIMS), desc='Simulating tournaments'):
    champ, round_reached = simulate_tournament()
    champion_counts[champ] = champion_counts.get(champ, 0) + 1
    for team, rnd in round_reached.items():
        if team not in round_counts:
            round_counts[team] = {}
        round_counts[team][rnd] = round_counts[team].get(rnd, 0) + 1

win_probs = pd.DataFrame([
    {'team': t, 'wins': c, 'win_pct': round(c / N_SIMS * 100, 2)}
    for t, c in champion_counts.items()
]).sort_values('win_pct', ascending=False).reset_index(drop=True)

print(f'Simulated {N_SIMS:,} tournaments')
print(f'Teams that won at least once: {len(win_probs)}')
print()
print('=== World Cup Winner Probabilities ===')
print(win_probs.head(20).to_string(index=False))
"""
]
nb['cells'][18]['outputs'] = []
nb['cells'][18]['execution_count'] = None

# Insert round analysis cell at position 19
round_analysis_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {'id': 'sim-round-analysis'},
    'outputs': [],
    'source': [
"""# Round reached analysis
ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]

elo_df_sim = pd.read_csv(PROCESSED_DIR / "final_elos.csv")
elo_map = dict(zip(elo_df_sim["team"], elo_df_sim["final_elo"]))

rows = []
for _, trow in teams_2026.iterrows():
    team = trow["team"]
    rc = round_counts.get(team, {"Group": N_SIMS})
    row = {
        "team": team,
        "group": trow["group"],
        "confederation": confs.get(team, "?"),
        "elo": round(elo_map.get(team, 0), 0),
    }
    for r in ROUND_ORDER:
        row[r] = round(rc.get(r, 0) / N_SIMS * 100, 1)
    row["win_pct"] = round(champion_counts.get(team, 0) / N_SIMS * 100, 2)
    rows.append(row)

round_df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)

print("=== Round Reached Analysis (% of 10,000 simulations) ===")
print()
print(f"{'Team':<22} {'ELO':>5} {'Out GS':>7} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Win':>6}")
print("-" * 82)
for _, r in round_df.iterrows():
    if r["elo"] < 1500: continue
    print(f"{r['team']:<22} {r['elo']:>5.0f} "
          f"{r['Group']:>6.1f}% {r['R32']:>5.1f}% {r['R16']:>5.1f}% "
          f"{r['QF']:>5.1f}% {r['SF']:>5.1f}% {r['Final']:>5.1f}% {r['win_pct']:>5.2f}%")

round_df.to_csv(PROCESSED_DIR / "wc2026_round_analysis.csv", index=False)
print("\\nSaved to wc2026_round_analysis.csv")
"""
    ]
}
nb['cells'].insert(19, round_analysis_cell)

with open('D:/project/fifa-2026-predictor/notebooks/07_tournament_simulation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Done. Total cells: {len(nb["cells"])}')
print('Updated: simulate_tournament (cell 15), MC cell (18), added round analysis (cell 19)')
