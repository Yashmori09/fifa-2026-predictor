"use client";

import { useState, useCallback, useRef } from "react";
import { TEAMS, GROUPS, getTeamsByGroup } from "@/lib/teams";

/* ──────────────────────── types ──────────────────────── */

interface GroupStanding {
  name: string;
  code: string;
  mp: number;
  w: number;
  d: number;
  l: number;
  gd: number;
  pts: number;
}

interface MatchResult {
  home: string;
  homeCode: string;
  away: string;
  awayCode: string;
  homeGoals: number;
  awayGoals: number;
  homeProb: number;
  drawProb: number;
  awayProb: number;
  group: string;
}

interface KnockoutMatch {
  id: string;
  round: string;
  home: string;
  homeCode: string;
  away: string;
  awayCode: string;
  homeGoals: number;
  awayGoals: number;
  homeProb: number;
  awayProb: number;
  winner: string;
  penalties: boolean;
}

type SimPhase = "idle" | "loading" | "group" | "r32" | "r16" | "qf" | "sf" | "final" | "done";
type Speed = "slow" | "fast" | "instant";

/* ──────────────────── static cards ──────────────────── */

const ROUND_CARDS = [
  { label: "GROUP STAGE", pct: "19.4%", team: "Spain", color: "text-purple" },
  { label: "ROUND OF 32", pct: "11.2%", team: "France", color: "text-purple" },
  { label: "ROUND OF 16", pct: "8.6%", team: "Argentina", color: "text-cyan" },
  { label: "QUARTER FINAL", pct: "8.1%", team: "England", color: "text-cyan" },
  { label: "SEMI FINAL", pct: "7.6%", team: "Mexico", color: "text-pink" },
  { label: "CHAMPION", pct: "19.4%", team: "Spain", color: "text-purple", highlight: true },
];

const GROUP_ELOS: Record<string, number> = {
  A: 1832, B: 1796, C: 1819, D: 1834, E: 1793, F: 1838,
  G: 1744, H: 1873, I: 1851, J: 1820, K: 1788, L: 1830,
};

/* ──────────── helpers ──────────── */

function getCode(name: string): string {
  return TEAMS.find((t) => t.name === name)?.code || "";
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ──────────── API response types ──────────── */

interface ApiGroupMatch {
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
  home_pts: number;
  away_pts: number;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
}

interface ApiGroupTeam {
  team: string;
  pts: number;
  gd: number;
  gf: number;
  ga: number;
  mp: number;
  w: number;
  d: number;
  l: number;
}

interface ApiGroup {
  group: string;
  matches: ApiGroupMatch[];
  table: ApiGroupTeam[];
}

interface ApiKnockoutMatch {
  team1: string;
  team2: string;
  team1_goals: number;
  team2_goals: number;
  winner: string;
  penalties: boolean;
  prob_team1: number;
  prob_draw: number;
  prob_team2: number;
}

interface ApiSimulateResponse {
  champion: string;
  groups: ApiGroup[];
  best_thirds: Record<string, string>;
  knockout: Record<string, ApiKnockoutMatch[]>;
}

/* ──────────────────────────── page ──────────────────────────── */

export default function TournamentPage() {
  const [phase, setPhase] = useState<SimPhase>("idle");
  const [speed, setSpeed] = useState<Speed>("fast");
  const [currentMatch, setCurrentMatch] = useState<MatchResult | null>(null);
  const [currentKOMatch, setCurrentKOMatch] = useState<KnockoutMatch | null>(null);
  const [groupMatchIndex, setGroupMatchIndex] = useState(0);

  const [standings, setStandings] = useState<Record<string, GroupStanding[]>>(() => {
    const s: Record<string, GroupStanding[]> = {};
    GROUPS.forEach((g) => {
      s[g] = getTeamsByGroup(g).map((t) => ({
        name: t.name, code: t.code, mp: 0, w: 0, d: 0, l: 0, gd: 0, pts: 0,
      }));
    });
    return s;
  });

  const [bracket, setBracket] = useState<Record<string, KnockoutMatch[]>>({
    r32: [], r16: [], qf: [], sf: [], final: [],
  });
  const [champion, setChampion] = useState<string | null>(null);

  const runningRef = useRef(false);
  const speedRef = useRef<Speed>(speed);
  speedRef.current = speed;

  const getDelay = useCallback(() => {
    const s = speedRef.current;
    if (s === "slow") return 1200;
    if (s === "fast") return 300;
    return 0;
  }, []);

  /* ────── convert API knockout match to our format ────── */
  const toKOMatch = (m: ApiKnockoutMatch, round: string, idx: number): KnockoutMatch => ({
    id: `${round}-${idx}`,
    round,
    home: m.team1,
    homeCode: getCode(m.team1),
    away: m.team2,
    awayCode: getCode(m.team2),
    homeGoals: m.team1_goals,
    awayGoals: m.team2_goals,
    homeProb: m.prob_team1,
    awayProb: m.prob_team2,
    winner: m.winner,
    penalties: m.penalties,
  });

  /* ────── animate group matches from pre-computed data ────── */
  const animateGroups = useCallback(async (groups: ApiGroup[]) => {
    setPhase("group");

    // Flatten all group matches in interleaved order (match-by-match across groups)
    // The API returns matches per group in fixture order, so we interleave them
    const maxMatches = Math.max(...groups.map((g) => g.matches.length));
    const allMatches: { match: ApiGroupMatch; group: string }[] = [];
    for (let i = 0; i < maxMatches; i++) {
      for (const g of groups) {
        if (i < g.matches.length) {
          allMatches.push({ match: g.matches[i], group: g.group });
        }
      }
    }

    // Progressive standings
    const live: Record<string, Record<string, { pts: number; gd: number; gf: number; ga: number; mp: number; w: number; d: number; l: number }>> = {};
    for (const g of groups) {
      live[g.group] = {};
      for (const t of g.table) {
        live[g.group][t.team] = { pts: 0, gd: 0, gf: 0, ga: 0, mp: 0, w: 0, d: 0, l: 0 };
      }
    }

    for (let i = 0; i < allMatches.length; i++) {
      if (!runningRef.current) return;
      const { match: m, group: gName } = allMatches[i];

      // Update live stats
      const h = live[gName][m.home_team];
      const a = live[gName][m.away_team];
      h.mp++; a.mp++;
      h.gf += m.home_goals; h.ga += m.away_goals;
      a.gf += m.away_goals; a.ga += m.home_goals;
      h.gd += m.home_goals - m.away_goals;
      a.gd += m.away_goals - m.home_goals;
      if (m.home_pts === 3) { h.w++; h.pts += 3; a.l++; }
      else if (m.away_pts === 3) { a.w++; a.pts += 3; h.l++; }
      else { h.d++; h.pts += 1; a.d++; a.pts += 1; }

      // Set current match display
      setCurrentMatch({
        home: m.home_team,
        homeCode: getCode(m.home_team),
        away: m.away_team,
        awayCode: getCode(m.away_team),
        homeGoals: m.home_goals,
        awayGoals: m.away_goals,
        homeProb: m.prob_home,
        drawProb: m.prob_draw,
        awayProb: m.prob_away,
        group: gName,
      });
      setGroupMatchIndex(i + 1);

      // Update standings for all groups
      setStandings(() => {
        const updated: Record<string, GroupStanding[]> = {};
        for (const g of groups) {
          const teamStats = Object.entries(live[g.group]).map(([team, stats]) => ({
            name: team,
            code: getCode(team),
            ...stats,
          }));
          teamStats.sort((x, y) => y.pts - x.pts || y.gd - x.gd || y.gf - x.gf);
          updated[g.group] = teamStats;
        }
        return updated;
      });

      if (getDelay() > 0) await sleep(getDelay());
    }
  }, [getDelay]);

  /* ────── animate knockout round from pre-computed data ────── */
  const animateKnockout = useCallback(async (
    roundName: string,
    matches: ApiKnockoutMatch[],
  ) => {
    setPhase(roundName as SimPhase);
    const converted: KnockoutMatch[] = [];

    for (let i = 0; i < matches.length; i++) {
      if (!runningRef.current) return;
      const km = toKOMatch(matches[i], roundName, i);
      converted.push(km);
      setCurrentKOMatch(km);
      setBracket((prev) => ({ ...prev, [roundName]: [...converted] }));
      if (getDelay() > 0) await sleep(getDelay());
    }
  }, [getDelay]);

  /* ────── START full simulation ────── */
  const startSimulation = useCallback(async () => {
    // Reset state
    setChampion(null);
    setCurrentMatch(null);
    setCurrentKOMatch(null);
    setGroupMatchIndex(0);
    setBracket({ r32: [], r16: [], qf: [], sf: [], final: [] });
    setStandings(() => {
      const s: Record<string, GroupStanding[]> = {};
      GROUPS.forEach((g) => {
        s[g] = getTeamsByGroup(g).map((t) => ({
          name: t.name, code: t.code, mp: 0, w: 0, d: 0, l: 0, gd: 0, pts: 0,
        }));
      });
      return s;
    });

    // Fetch from backend
    setPhase("loading");
    let data: ApiSimulateResponse;
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/simulate/full`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error("Simulation failed");
      data = await res.json();
    } catch {
      setPhase("idle");
      return;
    }

    runningRef.current = true;

    // Animate group stage
    await animateGroups(data.groups);
    if (!runningRef.current) return;

    // Animate knockout rounds
    const koRounds = ["r32", "r16", "qf", "sf", "final"] as const;
    for (const round of koRounds) {
      if (!data.knockout[round]) continue;
      await animateKnockout(round, data.knockout[round]);
      if (!runningRef.current) return;
    }

    setChampion(data.champion);
    setPhase("done");
    runningRef.current = false;
  }, [animateGroups, animateKnockout]);

  /* ────── skip to results (instant) ────── */
  const skipToResults = useCallback(() => {
    speedRef.current = "instant";
    setSpeed("instant");
  }, []);

  const isRunning = phase !== "idle" && phase !== "done" && phase !== "loading";

  /* ──────────────────────── render ──────────────────────── */
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center h-[160px] px-20 gap-2.5">
        <p className="text-cyan text-[11px] font-semibold tracking-[3px]">
          2026 FIFA WORLD CUP — USA · CANADA · MEXICO
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[48px] leading-tight">
          FULL TOURNAMENT BREAKDOWN
        </h1>
        <p className="text-secondary text-sm">
          48 teams · 12 groups · Round-by-round survival probabilities from
          10,000 simulations
        </p>
      </section>

      {/* Round-by-Round Survival */}
      <section className="px-20 py-10">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-1">
          ROUND-BY-ROUND SURVIVAL
        </h2>
        <p className="text-secondary text-xs mb-5">
          Highest survival probability at each stage from 10,000 Monte Carlo simulations
        </p>
        <div className="grid grid-cols-6 gap-3">
          {ROUND_CARDS.map((card) => (
            <div
              key={card.label}
              className={`flex flex-col items-center gap-2 bg-[#141414] rounded-lg px-4 py-5 border ${
                card.highlight ? "border-purple" : "border-[#1A1A1A]"
              }`}
            >
              <span className="text-secondary text-[10px] font-semibold tracking-[2px]">
                {card.label}
              </span>
              <span className={`font-[family-name:var(--font-anton)] text-[28px] ${card.color}`}>
                {card.pct}
              </span>
              <span className="text-xs">{card.team}</span>
            </div>
          ))}
        </div>
      </section>

      {/* All 12 Groups */}
      <section className="px-20 py-10">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
          ALL 12 GROUPS
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {GROUPS.map((group) => {
            const groupTeams = standings[group] || [];
            const avgElo = GROUP_ELOS[group] || 0;
            return (
              <div
                key={group}
                className="bg-surface border border-border rounded-lg overflow-hidden"
              >
                <div className="flex items-center justify-between px-3.5 h-9 bg-[#1A1A1A]">
                  <span className="font-[family-name:var(--font-anton)] text-[13px] tracking-wide">
                    GROUP {group}
                  </span>
                  <span className="font-mono text-[10px] text-secondary">
                    avg {avgElo}
                  </span>
                </div>
                {phase !== "idle" && phase !== "loading" && (
                  <div className="flex items-center justify-between px-3.5 h-7 border-b border-border text-[10px] text-secondary font-mono">
                    <span className="w-32">Team</span>
                    <span className="w-6 text-center">MP</span>
                    <span className="w-6 text-center">W</span>
                    <span className="w-6 text-center">D</span>
                    <span className="w-6 text-center">L</span>
                    <span className="w-8 text-center">GD</span>
                    <span className="w-8 text-center">PTS</span>
                  </div>
                )}
                {groupTeams.map((team, i) => (
                  <div
                    key={team.name}
                    className={`flex items-center justify-between px-3.5 h-9 ${
                      i < groupTeams.length - 1 ? "border-b border-border" : ""
                    } ${phase !== "idle" && phase !== "loading" && i < 2 ? "bg-[#141414]" : ""}`}
                  >
                    <div className="flex items-center gap-2 w-32">
                      <span className={`fi fi-${team.code}`} />
                      <span className="text-xs">{team.name}</span>
                    </div>
                    {phase !== "idle" && phase !== "loading" ? (
                      <>
                        <span className="font-mono text-[11px] text-secondary w-6 text-center">{team.mp}</span>
                        <span className="font-mono text-[11px] text-secondary w-6 text-center">{team.w}</span>
                        <span className="font-mono text-[11px] text-secondary w-6 text-center">{team.d}</span>
                        <span className="font-mono text-[11px] text-secondary w-6 text-center">{team.l}</span>
                        <span className={`font-mono text-[11px] w-8 text-center ${team.gd > 0 ? "text-cyan" : team.gd < 0 ? "text-pink" : "text-secondary"}`}>
                          {team.gd > 0 ? `+${team.gd}` : team.gd}
                        </span>
                        <span className={`font-mono text-[11px] font-semibold w-8 text-center ${i < 2 ? "text-purple" : "text-secondary"}`}>
                          {team.pts}
                        </span>
                      </>
                    ) : (
                      <span className={`font-mono text-[11px] ${i === 0 ? "text-purple" : "text-secondary"}`} />
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </section>

      {/* Simulation Controls */}
      <section className="px-20 py-8 flex items-center justify-center gap-6">
        {phase === "idle" && (
          <button
            onClick={startSimulation}
            className="px-10 py-3.5 rounded-lg bg-gradient-to-r from-purple to-pink text-white font-semibold text-sm tracking-wide hover:opacity-90 transition-opacity"
          >
            Start Simulation
          </button>
        )}
        {phase === "idle" && (
          <div className="flex items-center gap-2">
            {(["slow", "fast", "instant"] as Speed[]).map((s) => (
              <button
                key={s}
                onClick={() => { setSpeed(s); speedRef.current = s; }}
                className={`px-4 py-2 rounded text-xs font-mono border transition-colors ${
                  speed === s
                    ? "border-purple text-purple bg-purple/10"
                    : "border-border text-secondary hover:border-secondary"
                }`}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        )}
        {phase === "loading" && (
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="w-2 h-2 rounded-full bg-purple animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 rounded-full bg-cyan animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 rounded-full bg-pink animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-sm text-secondary font-mono">Running simulation on server...</span>
          </div>
        )}
        {isRunning && (
          <div className="flex items-center gap-4">
            <span className="text-sm text-secondary font-mono">
              {phase === "group"
                ? `Group Stage — Match ${groupMatchIndex}/72`
                : `${phase.toUpperCase()} Stage`}
            </span>
            <button
              onClick={skipToResults}
              className="px-6 py-2 rounded border border-border text-xs text-secondary hover:text-foreground hover:border-secondary transition-colors"
            >
              Skip to Results
            </button>
          </div>
        )}
        {phase === "done" && (
          <div className="flex items-center gap-6">
            {champion && (
              <div className="flex items-center gap-3">
                <span className={`fi fi-${getCode(champion)} text-3xl`} />
                <span className="font-[family-name:var(--font-anton)] text-2xl text-purple">
                  {champion}
                </span>
              </div>
            )}
            <button
              onClick={() => { setPhase("idle"); runningRef.current = false; }}
              className="px-8 py-3 rounded-lg bg-gradient-to-r from-purple to-pink text-white font-semibold text-sm tracking-wide hover:opacity-90 transition-opacity"
            >
              Simulate Again
            </button>
          </div>
        )}
      </section>

      {/* Live Match Card — Group Stage */}
      {phase === "group" && currentMatch && (
        <section className="px-20 py-8">
          <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-5">
            GROUP STAGE
          </h2>
          <div className="flex gap-6">
            <div className="flex-1 bg-[#111111] border border-border rounded-xl p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-secondary text-[10px] font-semibold tracking-[2px]">
                  GROUP {currentMatch.group}
                </span>
                <span className="text-[10px] text-secondary font-mono">
                  Match {groupMatchIndex}/72
                </span>
              </div>
              <div className="flex items-center justify-center gap-8 py-4">
                <div className="flex flex-col items-center gap-2">
                  <span className={`fi fi-${currentMatch.homeCode} text-4xl`} />
                  <span className="text-sm font-semibold">{currentMatch.home}</span>
                </div>
                <div className="font-[family-name:var(--font-anton)] text-[48px] tracking-wider">
                  {currentMatch.homeGoals} – {currentMatch.awayGoals}
                </div>
                <div className="flex flex-col items-center gap-2">
                  <span className={`fi fi-${currentMatch.awayCode} text-4xl`} />
                  <span className="text-sm font-semibold">{currentMatch.away}</span>
                </div>
              </div>
              <div className="flex gap-1 h-2 rounded-full overflow-hidden mt-2">
                <div className="bg-purple rounded-l-full" style={{ width: `${currentMatch.homeProb * 100}%` }} />
                <div className="bg-secondary" style={{ width: `${currentMatch.drawProb * 100}%` }} />
                <div className="bg-pink rounded-r-full" style={{ width: `${currentMatch.awayProb * 100}%` }} />
              </div>
              <div className="flex justify-between mt-1 text-[10px] font-mono text-secondary">
                <span>{(currentMatch.homeProb * 100).toFixed(0)}%</span>
                <span>Draw {(currentMatch.drawProb * 100).toFixed(0)}%</span>
                <span>{(currentMatch.awayProb * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="w-[340px] bg-[#111111] border border-border rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 h-9 bg-[#1A1A1A]">
                <span className="font-[family-name:var(--font-anton)] text-[13px] tracking-wide">
                  GROUP {currentMatch.group}
                </span>
                <span className="text-[10px] text-secondary font-mono">LIVE</span>
              </div>
              <div className="flex items-center px-4 h-7 border-b border-border text-[10px] text-secondary font-mono">
                <span className="flex-1">Team</span>
                <span className="w-6 text-center">MP</span>
                <span className="w-6 text-center">W</span>
                <span className="w-6 text-center">D</span>
                <span className="w-6 text-center">L</span>
                <span className="w-8 text-center">GD</span>
                <span className="w-8 text-center">PTS</span>
              </div>
              {(standings[currentMatch.group] || []).map((t, i) => (
                <div
                  key={t.name}
                  className={`flex items-center px-4 h-8 ${
                    i < 2 ? "bg-[#141414]" : ""
                  } ${i < 3 ? "border-b border-border" : ""}`}
                >
                  <div className="flex items-center gap-2 flex-1">
                    <span className={`fi fi-${t.code}`} />
                    <span className="text-xs">{t.name}</span>
                  </div>
                  <span className="font-mono text-[11px] text-secondary w-6 text-center">{t.mp}</span>
                  <span className="font-mono text-[11px] text-secondary w-6 text-center">{t.w}</span>
                  <span className="font-mono text-[11px] text-secondary w-6 text-center">{t.d}</span>
                  <span className="font-mono text-[11px] text-secondary w-6 text-center">{t.l}</span>
                  <span className={`font-mono text-[11px] w-8 text-center ${t.gd > 0 ? "text-cyan" : t.gd < 0 ? "text-pink" : "text-secondary"}`}>
                    {t.gd > 0 ? `+${t.gd}` : t.gd}
                  </span>
                  <span className={`font-mono text-[11px] font-semibold w-8 text-center ${i < 2 ? "text-purple" : "text-secondary"}`}>
                    {t.pts}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Live Match Card — Knockout */}
      {(phase === "r32" || phase === "r16" || phase === "qf" || phase === "sf" || phase === "final") && currentKOMatch && (
        <section className="px-20 py-8">
          <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-5">
            {phase === "r32" ? "ROUND OF 32" : phase === "r16" ? "ROUND OF 16" : phase === "qf" ? "QUARTER FINALS" : phase === "sf" ? "SEMI FINALS" : "FINAL"}
          </h2>
          <div className="max-w-xl mx-auto bg-[#111111] border border-border rounded-xl p-6">
            <div className="flex items-center justify-center gap-8 py-4">
              <div className="flex flex-col items-center gap-2">
                <span className={`fi fi-${currentKOMatch.homeCode} text-4xl`} />
                <span className={`text-sm font-semibold ${currentKOMatch.winner === currentKOMatch.home ? "text-purple" : ""}`}>
                  {currentKOMatch.home}
                </span>
              </div>
              <div className="flex flex-col items-center">
                <div className="font-[family-name:var(--font-anton)] text-[48px] tracking-wider">
                  {currentKOMatch.homeGoals} – {currentKOMatch.awayGoals}
                </div>
                {currentKOMatch.penalties && (
                  <span className="text-[10px] text-secondary font-mono">PEN</span>
                )}
              </div>
              <div className="flex flex-col items-center gap-2">
                <span className={`fi fi-${currentKOMatch.awayCode} text-4xl`} />
                <span className={`text-sm font-semibold ${currentKOMatch.winner === currentKOMatch.away ? "text-purple" : ""}`}>
                  {currentKOMatch.away}
                </span>
              </div>
            </div>
            <div className="flex gap-1 h-2 rounded-full overflow-hidden mt-2">
              <div className="bg-purple rounded-l-full" style={{ width: `${currentKOMatch.homeProb * 100}%` }} />
              <div className="bg-pink rounded-r-full" style={{ width: `${currentKOMatch.awayProb * 100}%` }} />
            </div>
            {currentKOMatch.winner && (
              <div className="text-center mt-3">
                <span className="text-xs text-purple font-semibold tracking-widest">
                  {currentKOMatch.winner} ADVANCES
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Knockout Bracket */}
      {(phase !== "idle" && phase !== "loading" && phase !== "group") && (
        <section className="px-20 py-10">
          <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
            KNOCKOUT BRACKET
          </h2>
          <div>
            <div className="flex gap-2 items-center">
              <BracketColumn matches={bracket.r32.slice(0, 8)} label="R32" />
              <BracketColumn matches={bracket.r16.slice(0, 4)} label="R16" />
              <BracketColumn matches={bracket.qf.slice(0, 2)} label="QF" />
              <BracketColumn matches={bracket.sf.slice(0, 1)} label="SF" />
              <div className="flex flex-col items-center gap-2">
                <span className="text-[10px] text-secondary font-semibold tracking-[2px] mb-2">FINAL</span>
                {bracket.final[0] ? (
                  <BracketMatch match={bracket.final[0]} isFinal />
                ) : (
                  <div className="w-[130px] h-[56px] bg-[#141414] border border-border rounded-lg flex items-center justify-center">
                    <span className="text-[10px] text-secondary">TBD</span>
                  </div>
                )}
                {champion && (
                  <div className="flex flex-col items-center gap-1 mt-2">
                    <span className={`fi fi-${getCode(champion)} text-2xl`} />
                    <span className="font-[family-name:var(--font-anton)] text-sm text-purple">{champion}</span>
                  </div>
                )}
              </div>
              <BracketColumn matches={bracket.sf.slice(1, 2)} label="SF" />
              <BracketColumn matches={bracket.qf.slice(2, 4)} label="QF" />
              <BracketColumn matches={bracket.r16.slice(4, 8)} label="R16" />
              <BracketColumn matches={bracket.r32.slice(8, 16)} label="R32" />
            </div>
          </div>
        </section>
      )}

      {/* Group Stage Complete Banner */}
      {phase !== "idle" && phase !== "loading" && phase !== "group" && (
        <section className="px-20 py-4">
          <div className="text-center py-3 bg-[#0D0D0D] border border-border rounded-lg">
            <span className="text-xs text-secondary font-mono">
              Group Stage Complete — 32 teams qualified for knockout rounds
            </span>
          </div>
        </section>
      )}
    </div>
  );
}

/* ──────────── bracket sub-components ──────────── */

function BracketColumn({ matches, label }: { matches: KnockoutMatch[]; label: string }) {
  return (
    <div className="flex flex-col items-center gap-3 flex-1">
      <span className="text-[10px] text-secondary font-semibold tracking-[2px] mb-2">
        {label}
      </span>
      {matches.length > 0 ? (
        matches.map((m) => <BracketMatch key={m.id} match={m} />)
      ) : (
        <div className="w-[130px] h-[56px] bg-[#141414] border border-border rounded-lg flex items-center justify-center">
          <span className="text-[10px] text-secondary">TBD</span>
        </div>
      )}
    </div>
  );
}

function BracketMatch({ match, isFinal }: { match: KnockoutMatch; isFinal?: boolean }) {
  const isDecided = match.winner !== "";
  return (
    <div className={`w-[130px] bg-[#141414] border rounded-lg overflow-hidden text-[11px] ${
      isFinal ? "border-purple" : "border-border"
    }`}>
      <div className={`flex items-center justify-between px-2.5 h-7 ${
        match.winner === match.home ? "bg-purple/10" : ""
      }`}>
        <div className="flex items-center gap-1.5">
          <span className={`fi fi-${match.homeCode} text-xs`} />
          <span className={`${match.winner === match.home ? "text-purple font-semibold" : "text-foreground"}`}>
            {match.home.length > 10 ? match.home.slice(0, 10) + "…" : match.home}
          </span>
        </div>
        {isDecided && <span className="font-mono font-semibold">{match.homeGoals}</span>}
      </div>
      <div className="h-px bg-border" />
      <div className={`flex items-center justify-between px-2.5 h-7 ${
        match.winner === match.away ? "bg-purple/10" : ""
      }`}>
        <div className="flex items-center gap-1.5">
          <span className={`fi fi-${match.awayCode} text-xs`} />
          <span className={`${match.winner === match.away ? "text-purple font-semibold" : "text-foreground"}`}>
            {match.away.length > 10 ? match.away.slice(0, 10) + "…" : match.away}
          </span>
        </div>
        {isDecided && <span className="font-mono font-semibold">{match.awayGoals}</span>}
      </div>
      {isDecided && match.penalties && (
        <div className="h-4 flex items-center justify-center bg-[#1A1A1A] border-t border-border">
          <span className="text-[8px] font-mono text-purple tracking-wider">PEN</span>
        </div>
      )}
    </div>
  );
}
