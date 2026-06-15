"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import { TEAMS } from "@/lib/teams";
import Tip from "@/components/Tip";
import type { LiveData, Match } from "@/lib/live-data";

/* ──────────────────── helpers ──────────────────── */

function teamCode(name: string | null): string {
  if (!name) return "xx";
  return TEAMS.find((t) => t.name === name)?.code ?? "xx";
}

// All times on the live page are rendered in IST (UTC+5:30, the project's home tz)
// regardless of the viewer's browser locale, so the day groupings line up.
const TZ = "Asia/Kolkata";

function formatKickoff(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  const date = d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: TZ }).toUpperCase();
  // hourCycle: "h23" gives 00-23 reliably across environments (hour12:false can yield 24:00 in some Node/Intl impls)
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: TZ });
  return { date, time };
}

function dayKey(iso: string): string {
  // en-CA gives YYYY-MM-DD, sortable + comparable
  return new Date(iso).toLocaleDateString("en-CA", { timeZone: TZ });
}

function dayLabel(iso: string): string {
  const matchKey = dayKey(iso);
  const now = new Date();
  const todayKey = now.toLocaleDateString("en-CA", { timeZone: TZ });
  const tomorrowKey = new Date(now.getTime() + 86_400_000).toLocaleDateString("en-CA", { timeZone: TZ });
  const yesterdayKey = new Date(now.getTime() - 86_400_000).toLocaleDateString("en-CA", { timeZone: TZ });
  if (matchKey === todayKey) return "TODAY";
  if (matchKey === tomorrowKey) return "TOMORROW";
  if (matchKey === yesterdayKey) return "YESTERDAY";
  return new Date(iso)
    .toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: TZ })
    .toUpperCase();
}

function pct(n: number | undefined): string {
  return n !== undefined ? `${Math.round(n * 100)}%` : "—";
}

function formatLastUpdated(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMin = Math.floor((now.getTime() - d.getTime()) / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin} min ago`;
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: TZ }) + " IST";
}

/* ──────────────── components ──────────────── */

function MatchRow({
  match,
  selected,
  onClick,
}: {
  match: Match;
  selected: boolean;
  onClick: () => void;
}) {
  const { time } = formatKickoff(match.kickoff_utc);
  const isLive = match.status === "IN_PLAY" || match.status === "PAUSED";
  const isFinished = match.status === "FINISHED";
  const homeCode = teamCode(match.home.name);
  const awayCode = teamCode(match.away.name);

  // Status indicators — color edge + right-side score by Match Score band
  let edgeColor = "bg-border";
  let statusIcon: React.ReactNode = null;
  if (isLive) {
    edgeColor = "bg-pink";
    statusIcon = <span className="text-[10px] text-pink font-mono font-bold tracking-wider animate-pulse">▶ LIVE</span>;
  } else if (isFinished && match.actual) {
    const ms = match.actual.match_score ?? 0;
    let textColor = "text-red-500";
    if (ms >= 80) { textColor = "text-green-500"; edgeColor = "bg-green-500"; }
    else if (ms >= 50) { textColor = "text-amber-500"; edgeColor = "bg-amber-500"; }
    else { edgeColor = "bg-red-500"; }
    statusIcon = (
      <div className="flex items-center gap-1">
        {match.actual.is_upset && <span className="text-pink text-[11px]" title="Upset">⚡</span>}
        <span className={`font-mono text-[11px] font-bold ${textColor}`}>{ms}</span>
      </div>
    );
  } else {
    statusIcon = <span className="text-secondary text-[10px] font-mono">IST</span>;
  }

  const homeScore = match.actual?.score.home ?? (isLive ? 0 : null);
  const awayScore = match.actual?.score.away ?? (isLive ? 0 : null);

  return (
    <button
      onClick={onClick}
      className={`w-full flex items-stretch gap-2.5 px-4 py-2 text-left border-b border-border transition-colors ${
        selected ? "bg-purple/15" : "hover:bg-[#1A1A1A]"
      }`}
    >
      <div className={`w-1 rounded ${edgeColor}`} />
      <div className="w-12 flex flex-col items-start gap-0.5 shrink-0">
        <span className={`text-[11px] font-mono font-semibold ${selected || isLive ? "text-foreground" : "text-secondary"}`}>
          {time}
        </span>
        <span className="text-[9px] font-mono font-semibold tracking-wider">
          {isFinished ? <span className="text-green-500">FT</span> : isLive ? <span className="text-pink">●</span> : <span className="text-secondary">IST</span>}
        </span>
      </div>
      <div className="flex-1 flex flex-col gap-0.5 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className={`fi fi-${homeCode} text-xs shrink-0`} />
          <span className={`text-[12px] truncate ${selected || isLive ? "text-foreground font-semibold" : "text-foreground/90"}`}>
            {match.home.name ?? "TBD"}
          </span>
          {homeScore !== null && (
            <span className="ml-auto font-mono text-[12px] font-bold text-foreground">{homeScore}</span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`fi fi-${awayCode} text-xs shrink-0`} />
          <span className={`text-[12px] truncate ${isFinished && match.actual?.outcome === "away" ? "text-foreground font-semibold" : "text-secondary"}`}>
            {match.away.name ?? "TBD"}
          </span>
          {awayScore !== null && (
            <span className="ml-auto font-mono text-[12px] font-bold text-secondary">{awayScore}</span>
          )}
        </div>
      </div>
      <div className="flex items-center pl-1">{statusIcon}</div>
    </button>
  );
}

function MatchDetail({ match }: { match: Match | null }) {
  if (!match) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#111111] border border-border rounded-2xl p-12 min-h-[400px]">
        <span className="text-secondary text-sm">Select a match from the list →</span>
      </div>
    );
  }

  const isLive = match.status === "IN_PLAY" || match.status === "PAUSED";
  const isFinished = match.status === "FINISHED";
  const homeCode = teamCode(match.home.name);
  const awayCode = teamCode(match.away.name);
  const { date, time } = formatKickoff(match.kickoff_utc);
  const p = match.prediction;
  const a = match.actual;

  const borderColor = isLive ? "border-pink" : isFinished && a?.outcome_correct ? "border-green-500" : isFinished ? "border-secondary/40" : "border-border";

  const liveOrFinishedScore = a ? (
    <div className="font-[family-name:var(--font-anton)] text-[72px] md:text-[96px] leading-none tracking-[6px]">
      <span className={a.outcome === "home" ? "text-foreground" : "text-secondary"}>{a.score.home}</span>
      <span className="text-secondary mx-2">–</span>
      <span className={a.outcome === "away" ? "text-foreground" : "text-secondary"}>{a.score.away}</span>
    </div>
  ) : isLive ? (
    <div className="font-[family-name:var(--font-anton)] text-[72px] md:text-[96px] leading-none tracking-[6px]">0 – 0</div>
  ) : (
    <div className="font-[family-name:var(--font-anton)] text-[40px] md:text-[56px] leading-none tracking-[6px] text-secondary">VS</div>
  );

  // Bar widths
  const hPct = p ? p.prob_home * 100 : 0;
  const dPct = p ? p.prob_draw * 100 : 0;
  const aPct = p ? p.prob_away * 100 : 0;

  return (
    <div className={`flex-1 flex flex-col gap-6 bg-[#111111] border ${borderColor} rounded-2xl p-7 md:p-8 transition-colors`}>
      {/* Status header */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          {isLive && (
            <div className="flex items-center gap-1.5 bg-pink px-2.5 py-1 rounded">
              <span className="w-1.5 h-1.5 rounded-full bg-white animate-livePulse" />
              <span className="text-[11px] font-mono font-bold tracking-[1.5px] text-white">LIVE</span>
            </div>
          )}
          {isFinished && a && (
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded ${
              a.outcome_correct ? "bg-green-500" : "bg-secondary/30"
            }`}>
              <span className="text-[11px] font-mono font-bold tracking-[1.5px] text-white">
                {a.outcome_correct ? "FINAL · CORRECT" : "FINAL"}
              </span>
            </div>
          )}
          <span className="text-[11px] font-mono font-semibold tracking-[1.5px] text-secondary">
            {match.stage === "group" ? `GROUP ${match.group}` : match.stage.toUpperCase()}
            {match.matchday && ` · MATCHDAY ${match.matchday}`}
          </span>
        </div>
        <span className="text-[11px] font-mono font-semibold tracking-[1.2px] text-secondary">
          {date} · {time} IST
        </span>
      </div>

      {/* Teams + score */}
      <div className="flex items-center justify-center gap-6 md:gap-12 py-2">
        <div className="flex-1 flex flex-col items-center gap-2">
          <span className={`fi fi-${homeCode} text-[64px] md:text-[84px]`} />
          <span className="font-[family-name:var(--font-anton)] text-lg md:text-2xl tracking-[2px] text-center">
            {match.home.name ?? "TBD"}
          </span>
        </div>
        <div className="flex flex-col items-center gap-2">
          {liveOrFinishedScore}
          {isLive && <span className="text-[10px] font-mono font-bold tracking-[2px] text-pink">LIVE</span>}
          {!isLive && !isFinished && p && (
            <span className="text-[10px] font-mono font-bold tracking-[2px] text-secondary">PRE-MATCH</span>
          )}
        </div>
        <div className="flex-1 flex flex-col items-center gap-2">
          <span className={`fi fi-${awayCode} text-[64px] md:text-[84px]`} />
          <span className="font-[family-name:var(--font-anton)] text-lg md:text-2xl tracking-[2px] text-center">
            {match.away.name ?? "TBD"}
          </span>
        </div>
      </div>

      {/* Prediction panel */}
      {p ? (
        <div className="flex flex-col gap-4 bg-[#0A0A0A] border border-border rounded-xl p-5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-mono font-bold tracking-[2px] text-secondary">OUR PRE-MATCH PREDICTION</span>
            <div className="flex items-center gap-2 bg-[#1A1A1A] rounded px-2 py-1">
              <span className="text-[10px]">🔒</span>
              <span className="text-[9px] font-mono font-bold tracking-[1.2px] text-secondary">LOCKED PRE-KICKOFF</span>
            </div>
          </div>

          {/* Probability values */}
          <div className="flex items-end justify-between gap-3">
            <div className="flex flex-col gap-0.5 items-start">
              <span className={`text-[10px] font-mono font-bold tracking-[1.5px] ${p.predicted_outcome === "home" ? "text-purple" : "text-secondary"}`}>
                {match.home.name?.toUpperCase().slice(0, 10) ?? "HOME"} WIN
              </span>
              <span className={`font-[family-name:var(--font-anton)] text-2xl md:text-3xl ${p.predicted_outcome === "home" ? "text-purple" : "text-secondary"}`}>
                {pct(p.prob_home)}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 items-center">
              <span className={`text-[10px] font-mono font-bold tracking-[1.5px] ${p.predicted_outcome === "draw" ? "text-cyan" : "text-secondary"}`}>DRAW</span>
              <span className={`font-[family-name:var(--font-anton)] text-xl md:text-2xl ${p.predicted_outcome === "draw" ? "text-cyan" : "text-secondary"}`}>
                {pct(p.prob_draw)}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 items-end">
              <span className={`text-[10px] font-mono font-bold tracking-[1.5px] ${p.predicted_outcome === "away" ? "text-pink" : "text-secondary"}`}>
                {match.away.name?.toUpperCase().slice(0, 10) ?? "AWAY"} WIN
              </span>
              <span className={`font-[family-name:var(--font-anton)] text-xl md:text-2xl ${p.predicted_outcome === "away" ? "text-pink" : "text-secondary"}`}>
                {pct(p.prob_away)}
              </span>
            </div>
          </div>

          {/* Probability bar */}
          <div className="flex gap-0.5 h-3.5 rounded-full overflow-hidden bg-[#1A1A1A]">
            <div className="bg-purple transition-all duration-700 ease-out" style={{ width: `${hPct}%` }} />
            <div className="bg-secondary transition-all duration-700 ease-out" style={{ width: `${dPct}%` }} />
            <div className="bg-pink transition-all duration-700 ease-out" style={{ width: `${aPct}%` }} />
          </div>

          {/* Predicted scoreline */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-mono font-bold tracking-[1.5px] text-secondary">PREDICTED SCORELINE</span>
              <span className="font-[family-name:var(--font-anton)] text-xl md:text-2xl tracking-[3px] text-cyan">
                {p.score.home} – {p.score.away}
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-[#0A0A0A] border border-border rounded-xl p-5 text-center">
          <span className="text-[12px] text-secondary">Prediction will lock once both teams are confirmed.</span>
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  sublabel,
  color,
  highlight,
}: {
  label: React.ReactNode;
  value: string;
  sublabel: string;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div className={`flex flex-col items-center gap-2 bg-[#111111] border ${highlight ? "border-green-500" : "border-[#1A1A1A]"} rounded-xl p-5 md:p-6`}>
      <span className={`text-[10px] font-mono font-bold tracking-[2px] ${color}`}>{label}</span>
      <span className="font-[family-name:var(--font-anton)] text-[40px] md:text-[48px] leading-none">{value}</span>
      <span className="text-[11px] font-mono text-secondary">{sublabel}</span>
    </div>
  );
}

function ResultCard({ match }: { match: Match }) {
  const p = match.prediction!;
  const a = match.actual!;
  const homeCode = teamCode(match.home.name);
  const awayCode = teamCode(match.away.name);
  const { date, time } = formatKickoff(match.kickoff_utc);

  const isUpset = a.is_upset;
  const matchScore = a.match_score ?? 0;
  const borderColor = isUpset ? "border-pink" : "border-border";
  const predTeamName = p.predicted_outcome === "home" ? match.home.name : p.predicted_outcome === "away" ? match.away.name : "Draw";
  const predProb = p.predicted_outcome === "home" ? p.prob_home : p.predicted_outcome === "away" ? p.prob_away : p.prob_draw;

  // Color the score badge: green 80+, amber 50-79, red <50
  let scoreClasses = "border-red-500 text-red-500";
  if (matchScore >= 80) scoreClasses = "border-green-500 text-green-500";
  else if (matchScore >= 50) scoreClasses = "border-amber-500 text-amber-500";

  return (
    <div className={`flex flex-col bg-[#111111] border ${borderColor} rounded-2xl overflow-hidden`}>
      {/* Top bar */}
      <div className="flex items-center justify-between gap-2 bg-[#0D0D0D] border-b border-border px-5 py-2.5 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-[11px] font-mono font-semibold tracking-[1.2px] text-secondary">{date} · {time} IST</span>
          <span className="text-[11px] font-mono font-semibold tracking-[1.2px] text-secondary">
            {match.stage === "group" ? `GROUP ${match.group}` : match.stage.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isUpset && (
            <div className="flex items-center gap-1.5 bg-pink rounded px-2.5 py-1">
              <span className="text-[10px]">⚡</span>
              <span className="text-[10px] font-mono font-bold tracking-[1.5px] text-white">UPSET</span>
            </div>
          )}
          <Tip term="Match Score">
            <div className={`flex items-baseline gap-1.5 bg-[#0A0A0A] border rounded px-2.5 py-1 ${scoreClasses}`}>
              <span className={`font-[family-name:var(--font-anton)] text-lg leading-none ${scoreClasses.split(" ")[1]}`}>
                {matchScore}
              </span>
              <span className="text-[9px] font-mono font-bold tracking-[1.5px] text-secondary">/ 100</span>
            </div>
          </Tip>
        </div>
      </div>

      {/* Body: split predicted | actual */}
      <div className="flex">
        {/* Predicted side */}
        <div className="flex-1 flex flex-col gap-3 bg-[#111111] border-r border-border p-5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-bold tracking-[2px] text-secondary">✦ WE PREDICTED</span>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <span className={`fi fi-${homeCode} text-xl opacity-60`} />
              <span className="text-[13px] font-semibold text-secondary truncate">{match.home.name}</span>
            </div>
            <span className="font-[family-name:var(--font-anton)] text-2xl md:text-3xl tracking-[3px] text-secondary">
              {p.score.home} – {p.score.away}
            </span>
            <div className="flex items-center gap-2">
              <span className={`fi fi-${awayCode} text-xl opacity-60`} />
              <span className="text-[13px] font-semibold text-secondary truncate">{match.away.name}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-secondary">{predTeamName === "Draw" ? "Draw" : `${predTeamName} win`}</span>
            <span className={`font-mono text-[13px] font-bold ${isUpset ? "text-pink" : "text-purple"}`}>
              {Math.round(predProb * 100)}%
            </span>
          </div>
        </div>

        {/* Actual side */}
        <div className="flex-1 flex flex-col gap-3 bg-[#0D0D0D] p-5">
          <div className="flex items-center gap-2">
            <span className={`text-[10px] font-mono font-bold tracking-[2px] ${matchScore >= 80 ? "text-green-500" : isUpset ? "text-pink" : "text-foreground"}`}>
              🏁 ACTUAL
            </span>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <span className={`fi fi-${homeCode} text-xl`} />
              <span className={`text-[13px] font-bold ${a.outcome === "home" ? "text-foreground" : "text-secondary"} truncate`}>
                {match.home.name}
              </span>
            </div>
            <span className="font-[family-name:var(--font-anton)] text-2xl md:text-3xl tracking-[3px] text-foreground">
              {a.score.home} – {a.score.away}
            </span>
            <div className="flex items-center gap-2">
              <span className={`fi fi-${awayCode} text-xl`} />
              <span className={`text-[13px] font-bold ${a.outcome === "away" ? "text-foreground" : "text-secondary"} truncate`}>
                {match.away.name}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-secondary">Goal-diff</span>
            <span className={`font-mono text-[13px] font-bold ${
              (a.goal_diff_error ?? 0) === 0 ? "text-green-500" :
              (a.goal_diff_error ?? 0) <= 1 ? "text-foreground" : "text-secondary"
            }`}>
              ±{a.goal_diff_error ?? 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ──────────────────────── page ──────────────────────── */

export default function LiveClient({ data: initialData }: { data: LiveData }) {
  const [data, setData] = useState<LiveData>(initialData);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const matches = data.matches;
  const stats = data.stats;

  // Auto-refresh: if any match is currently in play, poll the API route every 60s.
  // Otherwise just refetch when the tab regains focus (cheaper).
  const anyLive = useMemo(
    () => matches.some((m) => m.status === "IN_PLAY" || m.status === "PAUSED"),
    [matches]
  );

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        setIsRefreshing(true);
        const res = await fetch("/api/live-matches", { cache: "no-store" });
        if (!res.ok) return;
        const fresh = (await res.json()) as LiveData;
        if (!cancelled) setData(fresh);
      } catch {
        // silent
      } finally {
        if (!cancelled) setIsRefreshing(false);
      }
    }

    let intervalId: ReturnType<typeof setInterval> | null = null;
    if (anyLive) {
      intervalId = setInterval(refresh, 60_000);
    }
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
      window.removeEventListener("focus", onFocus);
    };
  }, [anyLive]);

  // Default selected: live match, else nearest upcoming, else last finished
  const defaultSelected = useMemo(() => {
    const live = matches.find((m) => m.status === "IN_PLAY" || m.status === "PAUSED");
    if (live) return live.id;
    const upcoming = matches.find((m) => m.status === "TIMED" || m.status === "SCHEDULED");
    if (upcoming) return upcoming.id;
    const last = [...matches].reverse().find((m) => m.status === "FINISHED");
    return last?.id ?? matches[0]?.id ?? 0;
  }, [matches]);

  const [selectedId, setSelectedId] = useState<number>(defaultSelected);
  const selected = useMemo(() => matches.find((m) => m.id === selectedId) ?? null, [matches, selectedId]);

  // Group matches by day for the list
  const groupedByDay = useMemo(() => {
    const groups: { day: string; label: string; matches: Match[] }[] = [];
    let currentDay = "";
    for (const m of matches) {
      const k = dayKey(m.kickoff_utc);
      if (k !== currentDay) {
        groups.push({ day: k, label: dayLabel(m.kickoff_utc), matches: [m] });
        currentDay = k;
      } else {
        groups[groups.length - 1].matches.push(m);
      }
    }
    return groups;
  }, [matches]);

  // Recent results (all finished, most recent first)
  const allResults = useMemo(() => {
    return matches
      .filter((m) => m.status === "FINISHED" && m.actual && m.prediction)
      .reverse();
  }, [matches]);

  // Filter pills
  type ResultFilter = "all" | "correct" | "wrong" | "upset";
  const [filter, setFilter] = useState<ResultFilter>("all");
  const [visibleCount, setVisibleCount] = useState(6);

  const filteredResults = useMemo(() => {
    if (filter === "all") return allResults;
    if (filter === "correct") return allResults.filter((m) => m.actual?.outcome_correct);
    if (filter === "wrong") return allResults.filter((m) => !m.actual?.outcome_correct);
    if (filter === "upset") return allResults.filter((m) => m.actual?.is_upset);
    return allResults;
  }, [allResults, filter]);

  // Reset visible count when filter changes
  useEffect(() => {
    setVisibleCount(6);
  }, [filter]);

  const visibleResults = filteredResults.slice(0, visibleCount);
  const hasMore = visibleCount < filteredResults.length;

  const filterCounts = useMemo(() => ({
    all: allResults.length,
    correct: allResults.filter((m) => m.actual?.outcome_correct).length,
    wrong: allResults.filter((m) => !m.actual?.outcome_correct).length,
    upset: allResults.filter((m) => m.actual?.is_upset).length,
  }), [allResults]);

  const nFinished = matches.filter((m) => m.status === "FINISHED").length;

  // Auto-scroll selected row into view in the list
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = document.getElementById(`row-${selectedId}`);
    if (el && listRef.current) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedId]);

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center px-4 md:px-12 lg:px-20 py-8 md:py-10 gap-2.5">
        <p className="text-cyan text-[11px] font-semibold tracking-[3px]">LIVE MATCH TRACKER</p>
        <h1 className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] lg:text-[56px] leading-tight">
          MATCH PREDICTIONS · LIVE
        </h1>
        <p className="text-secondary text-xs md:text-sm">
          Last updated {formatLastUpdated(data.last_updated)}{isRefreshing && " · refreshing…"} · {nFinished} of 104 matches played
        </p>
        <div className="w-[60px] h-1 bg-cyan rounded-sm mt-1" />
      </section>

      {/* Live & Upcoming */}
      <section className="px-4 md:px-12 lg:px-20 py-6 md:py-10">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-5">
          <div>
            <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[28px] tracking-wide">
              LIVE & UPCOMING
            </h2>
            <p className="text-secondary text-xs md:text-sm mt-1">
              Click any match to see how the model called it
            </p>
          </div>
          <div className="flex items-center gap-2 bg-[#111111] border border-border rounded px-3 py-1.5 w-fit">
            <span className={`w-1.5 h-1.5 rounded-full bg-pink ${anyLive ? "animate-livePulse" : ""}`} />
            <span className="text-[10px] font-mono font-bold tracking-[1.5px] text-secondary">
              {anyLive ? "AUTO-REFRESH · 60S" : "REFRESH ON VISIT"}
            </span>
          </div>
        </div>

        <div className="flex flex-col lg:flex-row gap-4">
          <MatchDetail match={selected} />

          {/* Match list sidebar */}
          <div className="w-full lg:w-[360px] flex flex-col bg-[#111111] border border-border rounded-2xl overflow-hidden shrink-0">
            <div className="flex items-center justify-between px-4 py-3 bg-[#0D0D0D] border-b border-border">
              <span className="text-[11px] font-mono font-bold tracking-[2px]">ALL MATCHES</span>
              <span className="text-[10px] font-mono font-semibold text-secondary">{nFinished} / 104</span>
            </div>
            <div ref={listRef} className="flex flex-col max-h-[640px] overflow-y-auto">
              {groupedByDay.map((group) => (
                <div key={group.day}>
                  <div className="sticky top-0 z-10 bg-[#0D0D0D] px-4 py-2 border-b border-border">
                    <span className={`text-[10px] font-mono font-bold tracking-[1.5px] ${
                      group.label === "TODAY" ? "text-cyan" : "text-secondary"
                    }`}>
                      {group.label}
                    </span>
                  </div>
                  {group.matches.map((m) => (
                    <div key={m.id} id={`row-${m.id}`}>
                      <MatchRow match={m} selected={selectedId === m.id} onClick={() => setSelectedId(m.id)} />
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Accuracy Dashboard */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-10 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[28px] tracking-wide mb-1">
          ACCURACY DASHBOARD
        </h2>
        <p className="text-secondary text-xs md:text-sm mb-6">
          How well our predictions match reality — updated as each match finishes
        </p>
        {stats.n_played === 0 ? (
          <p className="text-secondary text-sm">No matches scored yet — check back after the opening fixtures.</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <StatCard
              label={<Tip term="outcome accuracy">OUTCOME CORRECT</Tip>}
              color="text-purple"
              value={pct(stats.outcome_accuracy)}
              sublabel={`${stats.n_correct_outcome} of ${stats.n_played} matches`}
            />
            <StatCard
              label={<Tip term="Confidence Score">CONFIDENCE SCORE</Tip>}
              color="text-cyan"
              value={`${stats.avg_confidence_score ?? 0}`}
              sublabel="out of 100 · higher is better"
            />
            <StatCard
              label={<Tip term="Goal-Diff Error">GOAL-DIFF ERROR</Tip>}
              color="text-pink"
              value={`±${(stats.avg_goal_diff_error ?? 0).toFixed(1)}`}
              sublabel="avg goals off · lower is better"
            />
            <StatCard
              label={<Tip term="Upset">UPSETS CALLED</Tip>}
              color="text-green-500"
              highlight={(stats.n_upsets ?? 0) > 0}
              value={String(stats.n_upsets ?? 0)}
              sublabel={stats.biggest_upset ? `${stats.biggest_upset.winner} > ${stats.biggest_upset.fav_team}` : "No upsets yet"}
            />
          </div>
        )}
      </section>

      {/* Recent Results */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12">
        <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[28px] tracking-wide mb-1">
          RECENT RESULTS
        </h2>
        <p className="text-secondary text-xs md:text-sm mb-5">
          What we said before the whistle vs what actually happened
        </p>

        {/* Filter pills */}
        {allResults.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-5">
            {([
              { key: "all", label: "All", count: filterCounts.all, color: "border-foreground text-foreground" },
              { key: "correct", label: "Correct ✓", count: filterCounts.correct, color: "border-green-500 text-green-500" },
              { key: "wrong", label: "Wrong ✗", count: filterCounts.wrong, color: "border-secondary text-secondary" },
              { key: "upset", label: "Upsets ⚡", count: filterCounts.upset, color: "border-pink text-pink" },
            ] as { key: ResultFilter; label: string; count: number; color: string }[]).map((pill) => (
              <button
                key={pill.key}
                onClick={() => setFilter(pill.key)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-mono font-semibold tracking-wide border transition-colors ${
                  filter === pill.key
                    ? pill.color + " bg-[#111111]"
                    : "border-border text-secondary hover:border-secondary"
                }`}
              >
                <span>{pill.label}</span>
                <span className={`text-[10px] font-normal ${filter === pill.key ? "" : "text-secondary/60"}`}>{pill.count}</span>
              </button>
            ))}
          </div>
        )}

        {allResults.length === 0 ? (
          <p className="text-secondary text-sm">No completed matches yet.</p>
        ) : filteredResults.length === 0 ? (
          <p className="text-secondary text-sm">No matches match this filter yet.</p>
        ) : (
          <>
            <div className="flex flex-col gap-4">
              {visibleResults.map((m) => (
                <ResultCard key={m.id} match={m} />
              ))}
            </div>
            {hasMore && (
              <div className="flex justify-center mt-6">
                <button
                  onClick={() => setVisibleCount((c) => c + 6)}
                  className="px-6 py-2.5 rounded-full text-xs font-mono font-semibold tracking-wide border border-border text-secondary hover:border-purple hover:text-purple transition-colors"
                >
                  Load more · showing {visibleResults.length} of {filteredResults.length}
                </button>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
