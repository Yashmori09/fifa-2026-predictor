"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { TEAMS, GROUPS, getTeamsByGroup } from "@/lib/teams";
import { DETERMINISTIC_DATA } from "@/lib/deterministic-data";
import Tip from "@/components/Tip";

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
type SimMode = "prediction" | "simulation";

/* ──────────────────── static cards ──────────────────── */

const ROUND_CARDS = [
  { label: "GROUP STAGE", pct: "21.4%", team: "Spain", code: "es", color: "text-purple" },
  { label: "ROUND OF 32", pct: "21.0%", team: "France", code: "fr", color: "text-purple" },
  { label: "ROUND OF 16", pct: "10.8%", team: "Argentina", code: "ar", color: "text-cyan" },
  { label: "QUARTER FINAL", pct: "9.7%", team: "England", code: "gb-eng", color: "text-cyan" },
  { label: "SEMI FINAL", pct: "9.5%", team: "Germany", code: "de", color: "text-pink" },
  { label: "CHAMPION", pct: "21.4%", team: "Spain", code: "es", color: "text-purple", highlight: true },
];

const GROUP_ELOS: Record<string, number> = {
  A: 1832, B: 1796, C: 1819, D: 1834, E: 1793, F: 1838,
  G: 1744, H: 1873, I: 1851, J: 1820, K: 1788, L: 1830,
};

/* ──────────── squad quality (EA FC ratings) ──────────── */

const SQUAD_QUALITY: Record<string, number> = {
  France: 83.2, Spain: 82.1, England: 80.8, Germany: 80.7, Netherlands: 80.5,
  Portugal: 79.9, Argentina: 79.2, Brazil: 79.0, Belgium: 77.7, Turkey: 76.4,
  Croatia: 76.1, Switzerland: 75.8, "Ivory Coast": 74.9, Norway: 74.8,
  Uruguay: 74.6, Austria: 74.5, Senegal: 74.4, Colombia: 74.3,
  "Czech Republic": 74.3, "United States": 73.8, Sweden: 73.8, Morocco: 73.9,
  Scotland: 73.0, Japan: 72.2, Algeria: 71.8, Mexico: 71.7, Ghana: 71.3,
  "DR Congo": 71.2, Ecuador: 70.1, "Bosnia and Herzegovina": 70.0,
  Paraguay: 70.0, "South Korea": 69.4, Canada: 68.7, Uzbekistan: 68.7,
  Egypt: 68.0, "Cape Verde": 67.0, Australia: 66.4, Qatar: 66.4,
  "New Zealand": 65.6, Tunisia: 65.6, Haiti: 65.1, Iran: 64.0,
  Panama: 63.4, Jordan: 62.9, "Saudi Arabia": 62.3, Iraq: 62.0,
  "South Africa": 57.9, "Curaçao": 55.0,
};

interface FeaturedGroup {
  group: string;
  tag: string;
  tagColor: string;
  borderColor: string;
  avg: number;
  teams: { name: string; code: string; rating: number }[];
}

const FEATURED_GROUPS: FeaturedGroup[] = [
  {
    group: "L", tag: "#1 GROUP OF DEATH", tagColor: "text-red-500", borderColor: "border-red-500",
    avg: 72.9,
    teams: [
      { name: "England", code: "gb-eng", rating: 80.8 },
      { name: "Croatia", code: "hr", rating: 76.1 },
      { name: "Ghana", code: "gh", rating: 71.3 },
      { name: "Panama", code: "pa", rating: 63.4 },
    ],
  },
  {
    group: "K", tag: "#2 MOST COMPETITIVE", tagColor: "text-amber-500", borderColor: "border-amber-500",
    avg: 73.5,
    teams: [
      { name: "Portugal", code: "pt", rating: 79.9 },
      { name: "Colombia", code: "co", rating: 74.3 },
      { name: "DR Congo", code: "cd", rating: 71.2 },
      { name: "Uzbekistan", code: "uz", rating: 68.7 },
    ],
  },
  {
    group: "I", tag: "#3 STRONGEST FAVORITE", tagColor: "text-purple", borderColor: "border-purple",
    avg: 73.6,
    teams: [
      { name: "France", code: "fr", rating: 83.2 },
      { name: "Norway", code: "no", rating: 74.8 },
      { name: "Senegal", code: "sn", rating: 74.4 },
      { name: "Iraq", code: "iq", rating: 62.0 },
    ],
  },
];

const ALL_GROUPS_RANKED = [
  { group: "E", avg: 75.2 }, { group: "I", avg: 73.6 }, { group: "K", avg: 73.5 },
  { group: "F", avg: 73.0 }, { group: "L", avg: 72.9 }, { group: "C", avg: 72.8 },
  { group: "J", avg: 72.1 }, { group: "D", avg: 71.6 }, { group: "H", avg: 71.5 },
  { group: "B", avg: 70.2 }, { group: "G", avg: 68.8 }, { group: "A", avg: 68.3 },
];

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
  const [mode, setMode] = useState<SimMode>("prediction");
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

  const [hoveredGroup, setHoveredGroup] = useState<string | null>(null);
  const runningRef = useRef(false);
  const speedRef = useRef<Speed>(speed);
  speedRef.current = speed;

  // Auto-scroll refs
  const groupStageRef = useRef<HTMLElement>(null);
  const knockoutMatchRef = useRef<HTMLElement>(null);
  const bracketRef = useRef<HTMLElement>(null);
  const championRef = useRef<HTMLElement>(null);

  // Auto-scroll when phase changes
  useEffect(() => {
    const scrollTo = (ref: React.RefObject<HTMLElement | null>) => {
      setTimeout(() => {
        ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    };
    if (phase === "group") scrollTo(groupStageRef);
    else if (phase === "r32" || phase === "r16" || phase === "qf" || phase === "sf" || phase === "final") scrollTo(knockoutMatchRef);
    else if (phase === "done") scrollTo(championRef);
  }, [phase]);

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

    // Get data — either pre-computed or from API
    setHoveredGroup(null);
    let data: ApiSimulateResponse;
    if (mode === "prediction") {
      // Use pre-computed deterministic results (no API call)
      data = DETERMINISTIC_DATA as ApiSimulateResponse;
      setPhase("loading");
      // Brief loading state for visual continuity
      await sleep(400);
    } else {
      setPhase("loading");
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
  }, [mode, animateGroups, animateKnockout]);

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
      <section className="flex flex-col justify-center px-4 md:px-12 lg:px-20 py-8 md:py-10 gap-2.5">
        <p className="text-cyan text-[11px] font-semibold tracking-[3px]">
          2026 FIFA WORLD CUP — USA · CANADA · MEXICO
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[32px] md:text-[48px] leading-tight">
          FULL TOURNAMENT BREAKDOWN
        </h1>
        <p className="text-secondary text-sm">
          48 teams · 12 groups · Round-by-round survival probabilities from
          10,000 simulations
        </p>
      </section>

      {/* Round-by-Round Survival */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-10">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-1">
          ROUND-BY-ROUND SURVIVAL
        </h2>
        <p className="text-secondary text-xs mb-5">
          Highest survival probability at each stage — aggregated from 10,000 <Tip term="Monte Carlo">Monte Carlo</Tip> simulations, not a single run
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 md:gap-3">
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
              <div className="flex items-center gap-1.5">
                <span className={`fi fi-${card.code} text-base`} />
                <span className="text-xs">{card.team}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Group of Death */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[28px] tracking-wide mb-2">
          GROUP OF DEATH
        </h2>
        <p className="text-secondary text-sm leading-relaxed max-w-[700px] mb-7">
          Groups ranked by average squad quality (EA FC ratings). The tightest
          groups — where strong teams risk elimination — are the real danger
          zones.
        </p>

        {/* Featured groups */}
        <div className="flex flex-col md:flex-row gap-3 mb-8">
          {FEATURED_GROUPS.map((fg) => (
            <div
              key={fg.group}
              className={`flex-1 flex flex-col gap-3.5 bg-[#111111] border ${fg.borderColor} rounded-xl p-6`}
            >
              <span
                className={`font-mono text-[11px] font-semibold tracking-[2px] ${fg.tagColor}`}
              >
                {fg.tag}
              </span>
              <span className="font-[family-name:var(--font-anton)] text-2xl">
                Group {fg.group}
              </span>
              <span className="font-mono text-xs text-[#A1A1AA]">
                Avg <Tip term="EA FC">EA FC</Tip> Rating: {fg.avg}
              </span>
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-sm w-[18px]" />
                  <span className="text-[10px] text-secondary font-mono w-[100px]"></span>
                  <span className="flex-1" />
                  <span className="text-[10px] text-secondary font-mono w-10 text-right"><Tip term="EA FC">EA FC</Tip></span>
                </div>
                {fg.teams.map((t, i) => (
                  <div key={t.name} className="flex items-center gap-2">
                    <span className={`fi fi-${t.code} text-sm`} />
                    <span className="text-[13px] w-[100px]">{t.name}</span>
                    <div className="flex-1 h-2 bg-[#1A1A1A] rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          i === 0
                            ? "bg-purple"
                            : i === 1
                            ? "bg-cyan"
                            : "bg-[#2A2A2A]"
                        }`}
                        style={{
                          width: `${((t.rating - 50) / 40) * 100}%`,
                        }}
                      />
                    </div>
                    <span
                      className={`font-mono text-xs w-10 text-right ${
                        i === 0
                          ? "text-purple font-semibold"
                          : i === 1
                          ? "text-cyan"
                          : "text-secondary"
                      }`}
                      title="EA FC Rating"
                    >
                      {t.rating}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* All groups bar chart */}
        <p className="font-mono text-[11px] font-semibold tracking-[2px] text-secondary mb-4">
          ALL GROUPS BY AVG <Tip term="EA FC">EA FC</Tip> RATING
        </p>
        <div className="flex gap-2 items-end h-[120px]">
          {ALL_GROUPS_RANKED.map((g, i) => {
            const h = ((g.avg - 65) / 12) * 100;
            const isTop = i < 5;
            const isDeath = g.group === "L";
            return (
              <div
                key={g.group}
                className="flex-1 flex flex-col items-center gap-1 justify-end h-full"
              >
                <span
                  className={`font-[family-name:var(--font-anton)] text-base ${
                    isDeath ? "text-red-500" : "text-foreground"
                  }`}
                >
                  {g.group}
                </span>
                <div
                  className={`w-full rounded-t ${
                    isDeath
                      ? "bg-red-500"
                      : isTop
                      ? "bg-purple"
                      : "bg-[#2A2A2A]"
                  }`}
                  style={{ height: `${h}%` }}
                />
                <span className="font-mono text-[10px] text-[#A1A1AA]">
                  {g.avg}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      {/* All 12 Groups */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-10">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
          ALL 12 GROUPS
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {GROUPS.map((group) => {
            const groupTeams = standings[group] || [];
            const avgElo = GROUP_ELOS[group] || 0;
            const isHovered = hoveredGroup === group;
            const groupSquadAvg = ALL_GROUPS_RANKED.find((g) => g.group === group)?.avg || 0;
            return (
              <div
                key={group}
                className={`bg-surface border rounded-lg overflow-hidden transition-colors ${
                  isHovered ? "border-purple" : "border-border"
                }`}
                onMouseEnter={() => { if (phase === "idle") setHoveredGroup(group); }}
                onMouseLeave={() => setHoveredGroup(null)}
              >
                <div className="flex items-center justify-between px-3.5 h-9 bg-[#1A1A1A]">
                  <span className="font-[family-name:var(--font-anton)] text-[13px] tracking-wide">
                    GROUP {group}
                  </span>
                  <span className="font-mono text-[10px] text-secondary">
                    {isHovered ? <><Tip term="EA FC">EA FC</Tip> avg {groupSquadAvg}</> : <><Tip term="ELO">ELO</Tip> avg {avgElo}</>}
                  </span>
                </div>
                {phase !== "idle" && phase !== "loading" && !isHovered && (
                  <div className="flex items-center justify-between px-3.5 h-7 border-b border-border text-[10px] text-secondary font-mono">
                    <span className="w-32">Team</span>
                    <span className="w-6 text-center"><Tip term="MP">MP</Tip></span>
                    <span className="w-6 text-center"><Tip term="W">W</Tip></span>
                    <span className="w-6 text-center"><Tip term="D">D</Tip></span>
                    <span className="w-6 text-center"><Tip term="L">L</Tip></span>
                    <span className="w-8 text-center"><Tip term="GD">GD</Tip></span>
                    <span className="w-8 text-center"><Tip term="PTS">PTS</Tip></span>
                  </div>
                )}
                {isHovered && (
                  <div className="flex items-center justify-between px-3.5 h-7 border-b border-border text-[10px] text-secondary font-mono">
                    <span className="w-28">Team</span>
                    <span className="flex-1 text-center">Squad Rating</span>
                    <span className="w-10 text-right"><Tip term="EA FC">EA FC</Tip></span>
                  </div>
                )}
                {groupTeams.map((team, i) => {
                  const sq = SQUAD_QUALITY[team.name] || 0;
                  const barW = sq > 0 ? ((sq - 50) / 40) * 100 : 0;
                  return (
                    <div
                      key={team.name}
                      className={`flex items-center justify-between px-3.5 h-9 ${
                        i < groupTeams.length - 1 ? "border-b border-border" : ""
                      } ${!isHovered && phase !== "idle" && phase !== "loading" && i < 2 ? "bg-[#141414]" : ""}`}
                    >
                      <div className="flex items-center gap-2 w-28">
                        <span className={`fi fi-${team.code}`} />
                        <span className="text-xs">{team.name}</span>
                      </div>
                      {isHovered ? (
                        <>
                          <div className="flex-1 h-2 bg-[#1A1A1A] rounded-full overflow-hidden mx-2">
                            <div
                              className={`h-full rounded-full ${
                                i === 0 ? "bg-purple" : i === 1 ? "bg-cyan" : i === 2 ? "bg-pink" : "bg-amber-500"
                              }`}
                              style={{ width: `${barW}%` }}
                            />
                          </div>
                          <span
                            className={`font-mono text-[11px] w-10 text-right ${
                              i === 0 ? "text-purple font-semibold" : i === 1 ? "text-cyan" : i === 2 ? "text-pink" : "text-amber-500"
                            }`}
                          >
                            {sq > 0 ? sq.toFixed(1) : "—"}
                          </span>
                        </>
                      ) : phase !== "idle" && phase !== "loading" ? (
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
                  );
                })}
              </div>
            );
          })}
        </div>
      </section>

      {/* Simulation Section */}
      <section ref={championRef} className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        {phase === "idle" && (
          <div className="flex flex-col items-center gap-8 max-w-[800px] mx-auto">
            {/* Mode toggle */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMode("prediction")}
                className={`px-5 py-2.5 rounded-lg text-xs font-semibold tracking-wide border transition-all ${
                  mode === "prediction"
                    ? "border-purple bg-purple/10 text-purple"
                    : "border-border text-secondary hover:border-secondary hover:text-foreground"
                }`}
              >
                The Prediction
              </button>
              <button
                onClick={() => setMode("simulation")}
                className={`px-5 py-2.5 rounded-lg text-xs font-semibold tracking-wide border transition-all ${
                  mode === "simulation"
                    ? "border-cyan bg-cyan/10 text-cyan"
                    : "border-border text-secondary hover:border-secondary hover:text-foreground"
                }`}
              >
                What If?
              </button>
            </div>

            {mode === "prediction" ? (
              /* Prediction mode — simple prompt */
              <>
                <div className="text-center">
                  <p className="text-purple text-[11px] font-semibold tracking-[3px] mb-2">THE PREDICTION</p>
                  <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[32px] tracking-wide">
                    WHO DOES THE AI THINK WINS?
                  </h2>
                </div>
                <p className="text-[12px] text-secondary text-center max-w-[500px] leading-relaxed">
                  For every match, the AI picks the team most likely to win — no luck, no upsets. This is what the data says should happen.
                </p>
              </>
            ) : (
              /* Simulation mode — 3-step explainer */
              <>
                <div className="text-center">
                  <p className="text-cyan text-[11px] font-semibold tracking-[3px] mb-2">WHAT IF?</p>
                  <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[32px] tracking-wide">
                    WATCH ONE POSSIBLE WORLD CUP
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
                  <div className="flex flex-col items-center gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 text-center">
                    <span className="font-mono text-xs font-semibold text-purple tracking-wider">STEP 1</span>
                    <span className="text-sm font-bold">AI predicts every match</span>
                    <div className="flex items-center gap-2 bg-[#1A1A1A] rounded-lg px-3 py-2 w-full">
                      <span className="fi fi-ar text-base" />
                      <span className="text-[11px] flex-1 text-left">Argentina</span>
                      <span className="font-mono text-[11px] text-purple font-bold">65%</span>
                    </div>
                    <div className="flex items-center gap-2 bg-[#1A1A1A] rounded-lg px-3 py-2 w-full">
                      <span className="fi fi-mx text-base" />
                      <span className="text-[11px] flex-1 text-left">Mexico</span>
                      <span className="font-mono text-[11px] text-pink font-bold">15%</span>
                    </div>
                    <p className="text-[11px] text-[#A1A1AA] leading-relaxed">
                      The model gives win/draw/loss odds for all 104 matches
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 text-center">
                    <span className="font-mono text-xs font-semibold text-cyan tracking-wider">STEP 2</span>
                    <span className="text-sm font-bold">Randomness decides the result</span>
                    <div className="flex items-center justify-center gap-1 py-2 w-full">
                      <div className="flex gap-[2px]">
                        {Array.from({ length: 13 }).map((_, i) => (
                          <div key={`h${i}`} className="w-2 h-6 rounded-sm bg-purple" />
                        ))}
                        {Array.from({ length: 4 }).map((_, i) => (
                          <div key={`d${i}`} className="w-2 h-6 rounded-sm bg-secondary" />
                        ))}
                        {Array.from({ length: 3 }).map((_, i) => (
                          <div key={`a${i}`} className="w-2 h-6 rounded-sm bg-pink" />
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full bg-purple" />
                      <span className="text-[10px] text-secondary">65% Home</span>
                      <div className="w-2 h-2 rounded-full bg-secondary ml-2" />
                      <span className="text-[10px] text-secondary">20% Draw</span>
                      <div className="w-2 h-2 rounded-full bg-pink ml-2" />
                      <span className="text-[10px] text-secondary">15% Away</span>
                    </div>
                    <p className="text-[11px] text-[#A1A1AA] leading-relaxed">
                      Like picking a random ball — the favorite usually wins, but upsets happen
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 text-center">
                    <span className="font-mono text-xs font-semibold text-pink tracking-wider">STEP 3</span>
                    <span className="text-sm font-bold">A full World Cup plays out</span>
                    <div className="flex flex-col gap-1.5 w-full py-1">
                      <div className="flex items-center justify-between bg-[#1A1A1A] rounded px-2.5 py-1.5">
                        <span className="text-[11px]">Groups</span>
                        <span className="font-mono text-[10px] text-secondary">72 matches</span>
                      </div>
                      <div className="flex items-center justify-between bg-[#1A1A1A] rounded px-2.5 py-1.5">
                        <span className="text-[11px]">Knockouts</span>
                        <span className="font-mono text-[10px] text-secondary">32 matches</span>
                      </div>
                      <div className="flex items-center justify-between bg-purple/10 border border-purple/30 rounded px-2.5 py-1.5">
                        <span className="text-[11px] text-purple font-semibold">Champion</span>
                        <span className="font-mono text-[10px] text-purple">?</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-[#A1A1AA] leading-relaxed">
                      Every run produces a different winner — that&apos;s why it&apos;s a simulation, not a prediction
                    </p>
                  </div>
                </div>

                <p className="text-[11px] text-secondary text-center max-w-[500px]">
                  The probabilities on the home page come from running this 10,000 times. You&apos;re about to watch one.
                </p>
              </>
            )}

            {/* Controls */}
            <div className="flex flex-col md:flex-row items-center gap-4">
              <button
                onClick={startSimulation}
                className={`px-10 py-3.5 rounded-lg text-white font-semibold text-sm tracking-wide hover:opacity-90 transition-opacity ${
                  mode === "prediction"
                    ? "bg-gradient-to-r from-purple to-cyan"
                    : "bg-gradient-to-r from-purple to-pink"
                }`}
              >
                {mode === "prediction" ? "Reveal Winner" : "Roll the Dice"}
              </button>
              <div className="flex flex-col items-center gap-1.5">
                <span className="text-[10px] text-secondary font-mono tracking-wider">ANIMATION SPEED</span>
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
                      {s === "slow" ? "Slow" : s === "fast" ? "Fast" : "Skip"}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {phase === "loading" && (
          <div className="flex items-center justify-center gap-3 py-8">
            <div className="flex gap-1.5">
              <div className="w-2 h-2 rounded-full bg-purple animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 rounded-full bg-cyan animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 rounded-full bg-pink animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-sm text-secondary font-mono">Running simulation on server...</span>
          </div>
        )}

        {isRunning && (
          <div className="flex items-center justify-center gap-4 py-4">
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
          <div className="flex flex-col items-center gap-4 py-4">
            <p className="text-[10px] text-secondary font-mono tracking-wider">
              {mode === "prediction" ? "PREDICTED CHAMPION" : "THIS SIMULATION\u2019S CHAMPION"}
            </p>
            {champion && (
              <div className="flex items-center gap-4">
                <span className={`fi fi-${getCode(champion)} text-4xl`} />
                <span className="font-[family-name:var(--font-anton)] text-[32px] text-purple">
                  {champion}
                </span>
              </div>
            )}
            <p className="text-[12px] text-secondary text-center max-w-[400px]">
              {mode === "prediction"
                ? "The AI\u2019s best answer based on the data."
                : "This was one possible outcome. Try again \u2014 you\u2019ll likely get a different winner."}
            </p>
            <button
              onClick={() => { setPhase("idle"); runningRef.current = false; }}
              className={`px-8 py-3 rounded-lg text-white font-semibold text-sm tracking-wide hover:opacity-90 transition-opacity ${
                mode === "prediction"
                  ? "bg-gradient-to-r from-purple to-cyan"
                  : "bg-gradient-to-r from-purple to-pink"
              }`}
            >
              {mode === "prediction" ? "Watch Again" : "Roll Again"}
            </button>
          </div>
        )}
      </section>

      {/* Live Match Card — Group Stage */}
      {phase === "group" && currentMatch && (
        <section ref={groupStageRef} className="px-4 md:px-12 lg:px-20 py-6 md:py-8">
          <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-5">
            GROUP STAGE
          </h2>
          <div className="flex flex-col lg:flex-row gap-4 lg:gap-6">
            <div className="flex-1 bg-[#111111] border border-border rounded-xl p-4 md:p-6">
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
                <div className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] tracking-wider">
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

            <div className="w-full lg:w-[340px] bg-[#111111] border border-border rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 h-9 bg-[#1A1A1A]">
                <span className="font-[family-name:var(--font-anton)] text-[13px] tracking-wide">
                  GROUP {currentMatch.group}
                </span>
                <span className="text-[10px] text-secondary font-mono">LIVE</span>
              </div>
              <div className="flex items-center px-4 h-7 border-b border-border text-[10px] text-secondary font-mono">
                <span className="flex-1">Team</span>
                <span className="w-6 text-center"><Tip term="MP">MP</Tip></span>
                <span className="w-6 text-center"><Tip term="W">W</Tip></span>
                <span className="w-6 text-center"><Tip term="D">D</Tip></span>
                <span className="w-6 text-center"><Tip term="L">L</Tip></span>
                <span className="w-8 text-center"><Tip term="GD">GD</Tip></span>
                <span className="w-8 text-center"><Tip term="PTS">PTS</Tip></span>
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
        <section ref={knockoutMatchRef} className="px-4 md:px-12 lg:px-20 py-6 md:py-8">
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
                <div className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] tracking-wider">
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
        <section ref={bracketRef} className="px-4 md:px-12 lg:px-20 py-8 md:py-10">
          <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
            KNOCKOUT BRACKET
          </h2>
          <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0">
            <div className="flex gap-2 items-center min-w-[1100px]">
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
        <section className="px-4 md:px-12 lg:px-20 py-4">
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
