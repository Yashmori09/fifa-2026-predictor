"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { TEAMS } from "@/lib/teams";
import squadData from "@/data/squad_players.json";

/* ──────────────────── constants ──────────────────── */

const CONF_MAP: Record<string, string> = {
  Mexico: "CONCACAF", "South Korea": "AFC", "Czech Republic": "UEFA", "South Africa": "CAF",
  Canada: "CONCACAF", "Bosnia and Herzegovina": "UEFA", Qatar: "AFC", Switzerland: "UEFA",
  Brazil: "CONMEBOL", Morocco: "CAF", Haiti: "CONCACAF", Scotland: "UEFA",
  "United States": "CONCACAF", Paraguay: "CONMEBOL", Australia: "AFC", Turkey: "UEFA",
  Germany: "UEFA", "Curaçao": "CONCACAF", "Ivory Coast": "CAF", Ecuador: "CONMEBOL",
  Netherlands: "UEFA", Japan: "AFC", Sweden: "UEFA", Tunisia: "CAF",
  Belgium: "UEFA", Egypt: "CAF", Iran: "AFC", "New Zealand": "OFC",
  Spain: "UEFA", "Cape Verde": "CAF", "Saudi Arabia": "AFC", Uruguay: "CONMEBOL",
  France: "UEFA", Senegal: "CAF", Iraq: "AFC", Norway: "UEFA",
  Argentina: "CONMEBOL", Algeria: "CAF", Austria: "UEFA", Jordan: "AFC",
  Portugal: "UEFA", "DR Congo": "CAF", Uzbekistan: "AFC", Colombia: "CONMEBOL",
  England: "UEFA", Croatia: "UEFA", Ghana: "CAF", Panama: "CONCACAF",
};

/* ──────────── squad stats ──────────── */

interface PlayerData {
  name: string;
  pos: string;
  age: number;
  caps: number;
  goals: number;
  ovr: number;
  pot?: number;
  club: string;
}

interface TeamSquadStats {
  overall: number;
  gk: number;
  def: number;
  mid: number;
  fwd: number;
  top3: number;
  avgAge: number;
  topPlayers: PlayerData[];
}

const typedSquadData = squadData as Record<string, PlayerData[]>;

function getTeamStats(teamName: string): TeamSquadStats {
  const players = typedSquadData[teamName] ?? [];
  const byPos = (pos: string) => players.filter((p) => p.pos === pos).map((p) => p.ovr);
  const avg = (arr: number[]) =>
    arr.length ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : 0;
  const sorted = [...players].sort((a, b) => b.ovr - a.ovr);
  return {
    overall: avg(players.map((p) => p.ovr)),
    gk: avg(byPos("GK")),
    def: avg(byPos("DF")),
    mid: avg(byPos("MF")),
    fwd: avg(byPos("FW")),
    top3: avg(sorted.slice(0, 3).map((p) => p.ovr)),
    avgAge: avg(players.map((p) => p.age)),
    topPlayers: sorted.slice(0, 5),
  };
}

const POS_COLORS: Record<string, string> = {
  FW: "bg-pink text-white",
  MF: "bg-green-500 text-white",
  DF: "bg-cyan text-white",
  GK: "bg-purple text-white",
};

/* ── Animated counter ── */
function useCountUp(target: number, duration: number, active: boolean) {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number>(0);
  useEffect(() => {
    if (!active) { setValue(0); return; }
    const start = performance.now();
    const animate = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration, active]);
  return value;
}

/* ──────────── flow ──────────── */

type FlowPhase = "idle" | "loading" | "squad" | "stats" | "verdict" | "summary";
const PHASE_ORDER: FlowPhase[] = ["idle", "loading", "squad", "stats", "verdict", "summary"];

export default function PredictPage() {
  const [homeTeam, setHomeTeam] = useState("Spain");
  const [awayTeam, setAwayTeam] = useState("Argentina");
  const [predicting, setPredicting] = useState(false);
  const [showHomeSelector, setShowHomeSelector] = useState(false);
  const [showAwaySelector, setShowAwaySelector] = useState(false);
  const [result, setResult] = useState<{ home_win: number; draw: number; away_win: number; home_goals: number; away_goals: number } | null>(null);
  const [freshResult, setFreshResult] = useState(false);
  const [phase, setPhase] = useState<FlowPhase>("idle");
  const [statsAnimated, setStatsAnimated] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Section refs for auto-scroll
  const squadRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const verdictRef = useRef<HTMLDivElement>(null);
  const summaryRef = useRef<HTMLDivElement>(null);

  const homeData = TEAMS.find((t) => t.name === homeTeam);
  const awayData = TEAMS.find((t) => t.name === awayTeam);
  const homeStats = useMemo(() => getTeamStats(homeTeam), [homeTeam]);
  const awayStats = useMemo(() => getTeamStats(awayTeam), [awayTeam]);

  const clearTimers = () => { timerRef.current.forEach(clearTimeout); timerRef.current = []; };

  const scrollTo = useCallback((ref: React.RefObject<HTMLDivElement | null>) => {
    setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }, []);

  // Auto-scroll on phase change
  useEffect(() => {
    if (phase === "squad") scrollTo(squadRef);
    if (phase === "stats") { scrollTo(statsRef); setTimeout(() => setStatsAnimated(true), 200); }
    if (phase === "verdict") scrollTo(verdictRef);
    if (phase === "summary") scrollTo(summaryRef);
  }, [phase, scrollTo]);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;
    clearTimers();
    setPredicting(true);
    setResult(null);
    setFreshResult(false);
    setStatsAnimated(false);
    setPhase("loading");

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/predict/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult({ home_win: data.home_win, draw: data.draw, away_win: data.away_win, home_goals: data.home_goals, away_goals: data.away_goals });

      setPhase("squad");
      timerRef.current.push(setTimeout(() => setPhase("stats"), 2000));
      timerRef.current.push(setTimeout(() => setPhase("verdict"), 4000));
      timerRef.current.push(setTimeout(() => {
        setFreshResult(true);
        setPhase("summary");
        setPredicting(false);
      }, 6000));
    } catch {
      setResult(null);
      setPhase("idle");
      setPredicting(false);
    }
  };

  const resetFlow = () => {
    clearTimers();
    setResult(null);
    setFreshResult(false);
    setStatsAnimated(false);
    setPhase("idle");
    setPredicting(false);
  };

  const homeCount = useCountUp(result ? result.home_win * 100 : 0, 1000, freshResult);
  const drawCount = useCountUp(result ? result.draw * 100 : 0, 1000, freshResult);
  const awayCount = useCountUp(result ? result.away_win * 100 : 0, 1000, freshResult);
  const displayHome = freshResult ? homeCount : (result ? result.home_win * 100 : 0);
  const displayDraw = freshResult ? drawCount : (result ? result.draw * 100 : 0);
  const displayAway = freshResult ? awayCount : (result ? result.away_win * 100 : 0);

  const favored = result
    ? result.home_win >= result.away_win
      ? { name: homeTeam, code: homeData?.code, stats: homeStats, opp: awayStats, pct: displayHome }
      : { name: awayTeam, code: awayData?.code, stats: awayStats, opp: homeStats, pct: displayAway }
    : null;

  const advantages: string[] = [];
  if (favored) {
    const f = favored.stats, o = favored.opp;
    if (f.fwd > o.fwd) advantages.push(`Stronger attack — FWD avg ${f.fwd} vs ${o.fwd}`);
    if (f.mid > o.mid) advantages.push(`Better midfield — MID avg ${f.mid} vs ${o.mid}`);
    if (f.def > o.def) advantages.push(`Superior defense — DEF avg ${f.def} vs ${o.def}`);
    if (f.gk > o.gk) advantages.push(`Better goalkeeper — GK avg ${f.gk} vs ${o.gk}`);
    if (f.overall > o.overall) advantages.push(`Higher squad quality — Overall ${f.overall} vs ${o.overall}`);
    if (f.top3 > o.top3) advantages.push(`Better star players — Top 3 avg ${f.top3} vs ${o.top3}`);
  }

  const statRows = [
    { label: "OVERALL", home: homeStats.overall, away: awayStats.overall },
    { label: "ATTACK", home: homeStats.fwd, away: awayStats.fwd },
    { label: "MIDFIELD", home: homeStats.mid, away: awayStats.mid },
    { label: "DEFENSE", home: homeStats.def, away: awayStats.def },
    { label: "GK", home: homeStats.gk, away: awayStats.gk },
    { label: "STAR POWER", home: homeStats.top3, away: awayStats.top3 },
  ];

  const phaseGte = (p: FlowPhase) => PHASE_ORDER.indexOf(phase) >= PHASE_ORDER.indexOf(p);

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center px-4 md:px-12 lg:px-20 py-6 md:py-0 md:h-[120px] gap-2">
        <p className="text-pink text-[11px] font-semibold tracking-[3px]">AI MATCH PREDICTION</p>
        <h1 className="font-[family-name:var(--font-anton)] text-[32px] md:text-[48px] leading-tight">HEAD TO HEAD PREDICTOR</h1>
        <p className="text-secondary text-xs md:text-sm">Select any two nations to get instant AI-powered match probabilities</p>
      </section>

      {/* Team selector */}
      <section className="px-4 md:px-12 lg:px-20 py-8">
        <div className="flex flex-col items-center gap-6">
          <div className="flex flex-col md:flex-row items-center gap-4">
            {/* Home */}
            <div className={`flex flex-col items-center gap-4 bg-[#111111] border-2 rounded-xl px-6 md:px-8 py-7 md:py-9 w-full md:w-[240px] transition-all duration-300 ${
              predicting ? "border-purple shadow-[0_0_20px_rgba(168,85,247,0.3)]" : "border-purple"
            }`}>
              {homeData && <span className={`fi fi-${homeData.code} text-5xl md:text-6xl transition-transform duration-300 ${predicting ? "scale-110" : ""}`} />}
              <span className="font-[family-name:var(--font-anton)] text-[24px] md:text-[28px]">{homeTeam || "Select"}</span>
              <span className="font-mono text-[11px] text-secondary">{CONF_MAP[homeTeam] || "—"} · EA {homeStats.overall}</span>
              <div className="relative w-full">
                <button onClick={() => { setShowHomeSelector(!showHomeSelector); setShowAwaySelector(false); }} className="w-full h-9 bg-[#1A1A1A] rounded text-xs text-secondary hover:text-foreground transition-colors">Change Team</button>
                {showHomeSelector && (
                  <div className="absolute top-10 left-0 right-0 bg-[#1A1A1A] border border-border rounded-lg max-h-[240px] overflow-y-auto z-20">
                    {TEAMS.filter((t) => t.name !== awayTeam).map((t) => (
                      <button key={t.name} onClick={() => { setHomeTeam(t.name); setShowHomeSelector(false); resetFlow(); }} className="flex items-center gap-2 w-full px-3 py-2 text-xs text-left hover:bg-[#222] transition-colors">
                        <span className={`fi fi-${t.code}`} />{t.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* VS */}
            <div className="flex flex-col items-center gap-2 w-[60px]">
              <span className={`font-[family-name:var(--font-anton)] text-[32px] md:text-[44px] text-purple transition-transform duration-300 ${predicting ? "scale-125" : ""}`}>VS</span>
              <div className="hidden md:block w-0.5 h-[50px] bg-[#1A1A1A]" />
            </div>

            {/* Away */}
            <div className={`flex flex-col items-center gap-4 bg-[#111111] border-2 rounded-xl px-6 md:px-8 py-7 md:py-9 w-full md:w-[240px] transition-all duration-300 ${
              predicting ? "border-pink shadow-[0_0_20px_rgba(236,72,153,0.3)]" : "border-[#1A1A1A]"
            }`}>
              {awayData && <span className={`fi fi-${awayData.code} text-5xl md:text-6xl transition-transform duration-300 ${predicting ? "scale-110" : ""}`} />}
              <span className="font-[family-name:var(--font-anton)] text-[24px] md:text-[28px]">{awayTeam || "Select"}</span>
              <span className="font-mono text-[11px] text-secondary">{CONF_MAP[awayTeam] || "—"} · EA {awayStats.overall}</span>
              <div className="relative w-full">
                <button onClick={() => { setShowAwaySelector(!showAwaySelector); setShowHomeSelector(false); }} className="w-full h-9 bg-[#1A1A1A] rounded text-xs text-secondary hover:text-foreground transition-colors">Change Team</button>
                {showAwaySelector && (
                  <div className="absolute top-10 left-0 right-0 bg-[#1A1A1A] border border-border rounded-lg max-h-[240px] overflow-y-auto z-20">
                    {TEAMS.filter((t) => t.name !== homeTeam).map((t) => (
                      <button key={t.name} onClick={() => { setAwayTeam(t.name); setShowAwaySelector(false); resetFlow(); }} className="flex items-center gap-2 w-full px-3 py-2 text-xs text-left hover:bg-[#222] transition-colors">
                        <span className={`fi fi-${t.code}`} />{t.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={!homeTeam || !awayTeam || homeTeam === awayTeam || predicting}
            className={`w-full md:w-[560px] h-14 rounded-md bg-gradient-to-r from-purple to-pink text-white font-[family-name:var(--font-anton)] text-lg tracking-[3px] transition-all disabled:opacity-30 ${
              predicting ? "opacity-70 animate-pulse" : "hover:opacity-90"
            }`}
          >
            {predicting ? "PREDICTING..." : "PREDICT MATCH"}
          </button>
        </div>
      </section>

      {/* ──────── Flow Results ──────── */}
      {phase !== "idle" && (
        <section className="px-4 md:px-12 lg:px-20 pb-16 md:pb-20 flex flex-col gap-8 md:gap-10">

          {/* Loading */}
          {phase === "loading" && (
            <div className="flex flex-col items-center justify-center py-20 gap-4">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-purple animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-3 h-3 rounded-full bg-cyan animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-3 h-3 rounded-full bg-pink animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <span className="text-secondary text-sm">Analyzing squads and running prediction model...</span>
            </div>
          )}

          {/* ── Squad Reveal ── */}
          {phaseGte("squad") && phase !== "loading" && (
            <div ref={squadRef} className="flex flex-col gap-5 animate-fadeInUp scroll-mt-4">
              <div className="border-t border-border pt-6">
                <h3 className="font-[family-name:var(--font-anton)] text-[22px] tracking-wide">SQUAD REVEAL</h3>
              </div>
              <div className="flex flex-col md:flex-row gap-6 md:gap-8">
                {/* Home top 5 — slide from left */}
                <div className="flex-1 flex flex-col gap-2">
                  <span className="text-purple text-[13px] font-semibold">{homeTeam} — Top 5</span>
                  {homeStats.topPlayers.map((p, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 bg-[#1A1A1A] rounded-lg px-3 py-2.5 animate-slideLeft"
                      style={{ animationDelay: `${i * 120}ms` }}
                    >
                      <span className="text-[13px] font-semibold flex-1">{p.name}</span>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${POS_COLORS[p.pos] ?? "bg-[#2A2A2A] text-secondary"}`}>{p.pos}</span>
                      <span className="font-mono text-[13px] font-bold text-purple w-8 text-right">{p.ovr}</span>
                    </div>
                  ))}
                </div>
                {/* Away top 5 — slide from right */}
                <div className="flex-1 flex flex-col gap-2">
                  <span className="text-pink text-[13px] font-semibold">{awayTeam} — Top 5</span>
                  {awayStats.topPlayers.map((p, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 bg-[#1A1A1A] rounded-lg px-3 py-2.5 animate-slideRight"
                      style={{ animationDelay: `${i * 120}ms` }}
                    >
                      <span className="text-[13px] font-semibold flex-1">{p.name}</span>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${POS_COLORS[p.pos] ?? "bg-[#2A2A2A] text-secondary"}`}>{p.pos}</span>
                      <span className="font-mono text-[13px] font-bold text-pink w-8 text-right">{p.ovr}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Stat Comparison ── */}
          {phaseGte("stats") && phase !== "loading" && (
            <div ref={statsRef} className="flex flex-col gap-5 animate-fadeInUp scroll-mt-4">
              <div className="border-t border-border pt-6">
                <h3 className="font-[family-name:var(--font-anton)] text-[22px] tracking-wide">STAT COMPARISON</h3>
              </div>
              <div className="flex flex-col gap-3">
                {statRows.map((row, idx) => {
                  const homeW = row.home;
                  const awayW = row.away;
                  const homeWins = row.home > row.away;
                  const awayWins = row.away > row.home;
                  return (
                    <div
                      key={row.label}
                      className="flex items-center gap-3 animate-staggerFadeIn"
                      style={{ animationDelay: `${idx * 150}ms` }}
                    >
                      <span className={`font-mono text-sm font-bold w-12 text-right ${homeWins ? "text-purple" : "text-secondary"}`}>
                        {row.home}
                      </span>
                      <div className="flex-1 h-6 bg-[#1A1A1A] rounded overflow-hidden">
                        <div
                          className={`h-full rounded transition-all ease-out ${homeWins ? "bg-purple" : "bg-purple/40"} ${statsAnimated ? "animate-barGrow" : ""}`}
                          style={{
                            width: statsAnimated ? `${homeW}%` : "0%",
                            animationDelay: `${idx * 150}ms`,
                            animationDuration: "0.8s",
                          }}
                        />
                      </div>
                      <span className="text-[10px] md:text-[11px] font-semibold text-secondary tracking-wider w-16 md:w-24 text-center shrink-0">
                        {row.label}
                      </span>
                      <div className="flex-1 h-6 bg-[#1A1A1A] rounded overflow-hidden flex justify-end">
                        <div
                          className={`h-full rounded transition-all ease-out ${awayWins ? "bg-pink" : "bg-pink/40"} ${statsAnimated ? "animate-barGrow" : ""}`}
                          style={{
                            width: statsAnimated ? `${awayW}%` : "0%",
                            animationDelay: `${idx * 150}ms`,
                            animationDuration: "0.8s",
                          }}
                        />
                      </div>
                      <span className={`font-mono text-sm font-bold w-12 ${awayWins ? "text-pink" : "text-secondary"}`}>
                        {row.away}
                      </span>
                    </div>
                  );
                })}
              </div>
              {/* Score tally */}
              <div className="flex justify-center gap-8 mt-2">
                <span className="font-mono text-sm">
                  <span className="text-purple font-bold">{statRows.filter(r => r.home > r.away).length}</span>
                  <span className="text-secondary"> wins</span>
                </span>
                <span className="font-mono text-sm">
                  <span className="text-secondary font-bold">{statRows.filter(r => r.home === r.away).length}</span>
                  <span className="text-secondary"> tied</span>
                </span>
                <span className="font-mono text-sm">
                  <span className="text-pink font-bold">{statRows.filter(r => r.away > r.home).length}</span>
                  <span className="text-secondary"> wins</span>
                </span>
              </div>
            </div>
          )}

          {/* ── The Verdict ── */}
          {phaseGte("verdict") && phase !== "loading" && (
            <div ref={verdictRef} className="flex flex-col gap-5 animate-verdictReveal scroll-mt-4">
              <div className="border-t border-border pt-6">
                <h3 className="font-[family-name:var(--font-anton)] text-[22px] tracking-wide">THE VERDICT</h3>
              </div>
              <div className="flex flex-col gap-5 bg-[#111111] border border-[#1A1A1A] rounded-xl p-4 md:p-7 animate-glowPulse">
                <span className="text-secondary text-[13px]">{homeTeam} vs {awayTeam}</span>

                {/* Predicted Score */}
                {result && (
                  <div className="flex flex-col items-center gap-2 py-5 border border-border rounded-lg bg-[#0D0D0D]">
                    <span className="text-[10px] font-semibold tracking-[3px] text-secondary">PREDICTED SCORE</span>
                  <div className="flex items-center justify-center gap-6">
                    <div className="flex flex-col items-center gap-1">
                      {homeData && <span className={`fi fi-${homeData.code} text-2xl`} />}
                      <span className="text-[11px] text-secondary font-medium">{homeTeam}</span>
                    </div>
                    <div className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] tracking-wider">
                      <span className={result.home_goals > result.away_goals ? "text-purple" : "text-foreground"}>{result.home_goals}</span>
                      <span className="text-secondary mx-2">–</span>
                      <span className={result.away_goals > result.home_goals ? "text-pink" : "text-foreground"}>{result.away_goals}</span>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                      {awayData && <span className={`fi fi-${awayData.code} text-2xl`} />}
                      <span className="text-[11px] text-secondary font-medium">{awayTeam}</span>
                    </div>
                  </div>
                  {result.home_goals === result.away_goals && (
                    <span className="text-[11px] font-mono text-purple tracking-wider">
                      {result.home_win >= result.away_win ? homeTeam : awayTeam} wins on penalties
                    </span>
                  )}
                  </div>
                )}

                {/* Home win */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-sm">
                    <span><span className="text-purple font-mono text-xs">01</span> <span className="font-semibold">{homeTeam} Win</span></span>
                    <span className="font-mono text-cyan tabular-nums font-bold text-base">{displayHome.toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-[#1A1A1A] rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple to-cyan rounded-full transition-all duration-1000 ease-out" style={{ width: `${freshResult ? displayHome : (result?.home_win ?? 0) * 100}%` }} />
                  </div>
                </div>

                {/* Draw */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-sm">
                    <span className="font-semibold">Draw</span>
                    <span className="font-mono text-secondary tabular-nums">{displayDraw.toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-[#1A1A1A] rounded-full overflow-hidden">
                    <div className="h-full bg-secondary rounded-full transition-all duration-1000 ease-out" style={{ width: `${freshResult ? displayDraw : (result?.draw ?? 0) * 100}%` }} />
                  </div>
                </div>

                {/* Away win */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-sm">
                    <span><span className="text-pink font-mono text-xs">02</span> <span className="font-semibold">{awayTeam} Win</span></span>
                    <span className="font-mono text-pink tabular-nums font-bold text-base">{displayAway.toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-[#1A1A1A] rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-pink to-purple rounded-full transition-all duration-1000 ease-out" style={{ width: `${freshResult ? displayAway : (result?.away_win ?? 0) * 100}%` }} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Summary ── */}
          {phase === "summary" && favored && (
            <div ref={summaryRef} className="flex flex-col gap-5 animate-fadeInUp scroll-mt-4">
              <div className="border-t border-border pt-6">
                <h3 className="font-[family-name:var(--font-anton)] text-[22px] tracking-wide">
                  WHY {favored.name.toUpperCase()} IS FAVORED
                </h3>
              </div>
              <div className="flex flex-col gap-4 bg-[#111111] border border-green-500/30 rounded-xl p-4 md:p-7">
                <div className="flex items-center gap-4">
                  {favored.code && <span className={`fi fi-${favored.code} text-4xl`} />}
                  <div>
                    <span className="font-[family-name:var(--font-anton)] text-[28px] text-green-400">
                      {favored.pct.toFixed(1)}%
                    </span>
                    <p className="text-secondary text-[13px]">chance of winning this match</p>
                  </div>
                </div>
                {advantages.length > 0 ? (
                  <div className="flex flex-col gap-2.5">
                    {advantages.map((a, i) => (
                      <span
                        key={i}
                        className="text-[13px] text-[#A1A1AA] animate-staggerFadeIn"
                        style={{ animationDelay: `${i * 100}ms` }}
                      >
                        ✦ {a}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-[13px] text-[#A1A1AA]">
                    Close matchup — the model gives a slight edge based on combined features
                  </span>
                )}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
