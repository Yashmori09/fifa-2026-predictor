"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import type { RetroData, RetroTeam } from "./page";

const VERDICT_COLOR: Record<string, string> = {
  overachieved: "text-cyan",
  matched: "text-purple",
  underperformed: "text-pink",
};

const VERDICT_TAG: Record<string, string> = {
  overachieved: "OVERACHIEVED",
  matched: "AS PREDICTED",
  underperformed: "UNDERPERFORMED",
};

const VERDICT_EMOJI: Record<string, string> = {
  overachieved: "🚀",
  matched: "🎯",
  underperformed: "💔",
};

function pct(n: number | undefined): string {
  if (n === undefined) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

function pct4(n: number): string {
  return `${(n * 100).toFixed(2)}%`;
}

/* ─────────── team story component ─────────── */

function TeamStory({ team, champion }: { team: RetroTeam; champion: string }) {
  const isChampion = team.actual.finish_stage === "champion";
  const isVillain = team.team === champion;
  const wasBeatenByChampion = team.elimination?.opponent === champion;

  return (
    <div key={team.team} className="flex flex-col gap-8 mt-8 animate-[fadeInUp_0.5s_ease-out]">
      {/* Chapter 1 — What we said */}
      <section className="bg-surface border border-border rounded-xl p-5 md:p-7">
        <p className="text-purple text-[10px] font-semibold tracking-[3px] mb-2">CHAPTER 1</p>
        <h3 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
          WHAT WE PREDICTED
        </h3>
        <div className="text-[13px] md:text-[15px] leading-relaxed text-secondary">
          We gave{" "}
          <span className="text-foreground font-semibold">{team.team}</span>{" "}
          a <span className="text-purple font-semibold">{pct4(team.predicted.p_win)}</span>{" "}
          chance to win the tournament. Our aggregate expected finish:{" "}
          <span className="text-foreground font-semibold">{team.predicted.expected_finish}</span>
          .
        </div>

        <div className="mt-5 grid grid-cols-5 gap-2 md:gap-4">
          {[
            { label: "Reach R16", val: team.predicted.p_r16 },
            { label: "Reach QF",  val: team.predicted.p_qf },
            { label: "Reach SF",  val: team.predicted.p_sf },
            { label: "Reach F",   val: team.predicted.p_final },
            { label: "Win it",    val: team.predicted.p_win },
          ].map((r) => (
            <div key={r.label} className="flex flex-col items-center gap-1">
              <span className="font-mono text-[10px] text-secondary">{r.label}</span>
              <span className="font-[family-name:var(--font-anton)] text-[16px] md:text-[20px] text-cyan">
                {pct(r.val)}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter 2 — What happened */}
      <section className="bg-surface border border-border rounded-xl p-5 md:p-7">
        <p className="text-purple text-[10px] font-semibold tracking-[3px] mb-2">CHAPTER 2</p>
        <h3 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
          WHAT HAPPENED
        </h3>
        <div className="flex items-baseline gap-4 mb-4">
          <span className="text-4xl md:text-5xl">{VERDICT_EMOJI[team.verdict]}</span>
          <div className="flex flex-col">
            <span className={`font-mono text-[10px] font-semibold tracking-[2px] ${VERDICT_COLOR[team.verdict]}`}>
              {VERDICT_TAG[team.verdict]}
            </span>
            <span className="font-[family-name:var(--font-anton)] text-[22px] md:text-[30px]">
              {team.actual.finish_label.toUpperCase()}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-4">
          {[
            { label: "MP", val: team.actual.matches_played },
            { label: "W",  val: team.actual.wins },
            { label: "D",  val: team.actual.draws },
            { label: "L",  val: team.actual.losses },
            { label: "GF", val: team.actual.gf },
            { label: "GA", val: team.actual.ga },
          ].map((s) => (
            <div key={s.label} className="flex flex-col items-center gap-1 bg-background rounded-lg py-2">
              <span className="font-mono text-[10px] text-secondary">{s.label}</span>
              <span className="font-[family-name:var(--font-anton)] text-[20px]">{s.val}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter 3 — The Elimination (only if not champion) */}
      {team.elimination && !isChampion && (
        <section className="bg-surface border border-border rounded-xl p-5 md:p-7">
          <p className="text-purple text-[10px] font-semibold tracking-[3px] mb-2">CHAPTER 3</p>
          <h3 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
            {isVillain ? "THE FINAL LOSS" : "WHO ELIMINATED THEM"}
          </h3>

          <div className="flex items-center gap-4 md:gap-6 mb-4">
            <div className="flex flex-col items-center gap-1 flex-1">
              <span className={`fi fi-${team.flag} text-3xl md:text-4xl`} />
              <span className="text-[13px] md:text-[15px] font-semibold">{team.team}</span>
            </div>
            <div className="font-[family-name:var(--font-anton)] text-[32px] md:text-[44px] text-secondary">
              {team.elimination.score}
            </div>
            <div className="flex flex-col items-center gap-1 flex-1">
              <span className={`fi fi-${team.elimination.opponent_flag ?? "xx"} text-3xl md:text-4xl`} />
              <span className="text-[13px] md:text-[15px] font-semibold">{team.elimination.opponent}</span>
            </div>
          </div>

          <p className="text-secondary text-[13px] md:text-[15px] leading-relaxed mb-3">
            {team.team} lost to <span className="text-foreground font-semibold">{team.elimination.opponent}</span>{" "}
            in the <span className="text-foreground">{team.elimination.stage_label}</span>.
            {team.elimination.our_win_prob !== undefined && (
              <>
                {" "}Our pre-match model called this{" "}
                <span className="font-mono text-cyan">
                  {pct(team.elimination.our_win_prob)}
                </span>
                {" / "}
                <span className="font-mono text-secondary">{pct(team.elimination.draw_prob)}</span>
                {" / "}
                <span className="font-mono text-pink">
                  {pct(team.elimination.opp_win_prob)}
                </span>{" "}
                (win/draw/loss).{" "}
                {team.elimination.our_pick === "loss" ? (
                  <span className="text-secondary">The model called it — we favored their opponent.</span>
                ) : team.elimination.our_pick === "win" ? (
                  <span className="text-pink">We had them winning this one. Ouch.</span>
                ) : (
                  <span className="text-secondary">We called it a coin flip.</span>
                )}
              </>
            )}
          </p>

          {wasBeatenByChampion && (
            <div className="mt-4 p-3 bg-purple/10 border border-purple/30 rounded-lg">
              <p className="text-[12px] text-purple font-mono">
                → Your villain went on to win the whole tournament. So there&apos;s that.
              </p>
            </div>
          )}
        </section>
      )}

      {/* Chapter 3 (champion variant) */}
      {isChampion && (
        <section className="bg-gradient-to-br from-purple/20 to-cyan/10 border border-purple/40 rounded-xl p-5 md:p-7">
          <p className="text-purple text-[10px] font-semibold tracking-[3px] mb-2">CHAPTER 3</p>
          <h3 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
            🏆 CHAMPIONS
          </h3>
          <p className="text-[13px] md:text-[15px] leading-relaxed text-secondary">
            {team.team} won the World Cup. They played{" "}
            <span className="text-foreground font-semibold">{team.actual.matches_played} matches</span>,
            won <span className="text-foreground font-semibold">{team.actual.wins}</span>
            {team.actual.losses > 0
              ? <>, and lost <span className="text-pink font-semibold">{team.actual.losses}</span>.</>
              : <> without losing a single one.</>
            }
          </p>
        </section>
      )}

      {/* Chapter 4 — Alternative universe */}
      <section className="bg-surface border border-border rounded-xl p-5 md:p-7">
        <p className="text-purple text-[10px] font-semibold tracking-[3px] mb-2">CHAPTER 4</p>
        <h3 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
          ALTERNATIVE UNIVERSES
        </h3>
        <p className="text-secondary text-[13px] md:text-[15px] leading-relaxed">
          In our 100,000 Monte Carlo simulations,{" "}
          <span className="text-foreground font-semibold">{team.team}</span> won the tournament in{" "}
          <span className="text-cyan font-mono font-semibold">
            {Math.round(team.predicted.p_win * 100000).toLocaleString()}
          </span>{" "}
          of them ({pct4(team.predicted.p_win)}). They reached the Final in{" "}
          <span className="text-purple font-mono">{Math.round(team.predicted.p_final * 100000).toLocaleString()}</span>{" "}
          alternate timelines, and made the semis in{" "}
          <span className="text-purple font-mono">{Math.round(team.predicted.p_sf * 100000).toLocaleString()}</span>.
        </p>

        <div className="mt-4 p-3 bg-background rounded-lg font-mono text-[11px] text-secondary leading-relaxed">
          {isChampion ? (
            <>
              The model gave you a <span className="text-cyan">{pct4(team.predicted.p_win)}</span> chance.
              You beat those odds. In {Math.round((1 - team.predicted.p_win) * 100)}% of universes, someone else won.
              This is one of the {Math.round(team.predicted.p_win * 100)}% where you did.
            </>
          ) : team.verdict === "overachieved" ? (
            <>
              Reality landed in one of the more optimistic branches of our simulation for {team.team}.
              The average tournament had them finishing earlier than they actually did.
            </>
          ) : team.verdict === "underperformed" ? (
            <>
              Reality landed in one of the more pessimistic branches of our simulation for {team.team}.
              In most alternative universes, they went further than they did.
            </>
          ) : (
            <>
              Reality tracked closely with the median outcome of our simulation for {team.team}.
              The model and reality agreed on this one.
            </>
          )}
        </div>
      </section>
    </div>
  );
}

/* ─────────── main client ─────────── */

export default function RetrospectiveClient({ data }: { data: RetroData }) {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const storyRef = useRef<HTMLDivElement>(null);

  // Sort teams alphabetically for the picker
  const sortedTeams = useMemo(
    () => Object.values(data.teams).sort((a, b) => a.team.localeCompare(b.team)),
    [data.teams]
  );

  // When a team is picked, scroll to the story
  useEffect(() => {
    if (selectedTeam && storyRef.current) {
      storyRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedTeam]);

  const team = selectedTeam ? data.teams[selectedTeam] : null;

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center min-h-[200px] md:min-h-[280px] px-4 md:px-12 lg:px-20 py-8 md:py-0 gap-3 md:gap-4">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          WC 2026 · POST-TOURNAMENT RETROSPECTIVE
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[40px] md:text-[56px] lg:text-[72px] leading-[1] max-w-[860px]">
          WHO BEAT
          <br />
          YOUR TEAM?
        </h1>
        <p className="text-secondary text-[13px] md:text-[15px] max-w-[820px]">
          The tournament is over. <span className="text-foreground font-semibold">{data.overall.champion}</span> won,{" "}
          <span className="text-foreground font-semibold">{data.overall.runner_up}</span> runner-up. Pick your team below to
          see what we predicted, what actually happened, and who eliminated them.
        </p>
        <div className="w-[60px] h-1 bg-purple rounded-sm" />
      </section>

      {/* Team grid picker */}
      <section className="px-4 md:px-12 lg:px-20 py-6 md:py-8">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between mb-4 md:mb-6 gap-1">
          <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide">
            PICK YOUR TEAM
          </h2>
          <p className="text-secondary text-[11px]">All 48 nations · click to reveal story</p>
        </div>

        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-2 md:gap-3">
          {sortedTeams.map((t) => {
            const active = selectedTeam === t.team;
            return (
              <button
                key={t.team}
                onClick={() => setSelectedTeam(t.team)}
                className={`flex flex-col items-center gap-1 md:gap-1.5 py-2 md:py-3 px-1 rounded-lg border transition-all ${
                  active
                    ? "border-purple bg-purple/10 scale-[1.02]"
                    : "border-border bg-surface hover:border-purple/50 hover:bg-purple/5"
                }`}
                aria-label={`Pick ${t.team}`}
              >
                <span className={`fi fi-${t.flag} text-lg md:text-2xl`} />
                <span className={`text-[9px] md:text-[10px] font-mono text-center leading-tight ${
                  active ? "text-foreground" : "text-secondary"
                }`}>
                  {t.team.length > 12 ? t.team.slice(0, 10) + "…" : t.team}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Selected team story */}
      <section ref={storyRef} className="px-4 md:px-12 lg:px-20 py-6 md:py-8">
        {team ? (
          <TeamStory team={team} champion={data.overall.champion} />
        ) : (
          <div className="text-center py-12 text-secondary text-[13px]">
            ↑ Pick a team to see their story
          </div>
        )}
      </section>

      {/* Overall stats footer */}
      <section className="px-4 md:px-12 lg:px-20 py-10 md:py-16 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-4">
          HOW THE MODEL DID
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Overall accuracy", val: `${((data.overall.stats.outcome_accuracy ?? 0) * 100).toFixed(1)}%` },
            { label: "Matches played", val: `${data.overall.stats.n_played}` },
            { label: "Confidence score", val: `${(data.overall.stats.avg_confidence_score ?? 0).toFixed(0)}/100` },
            { label: "Upsets", val: `${data.overall.stats.n_upsets ?? 0}` },
          ].map((s) => (
            <div key={s.label} className="bg-surface border border-border rounded-lg p-4">
              <p className="font-mono text-[10px] text-secondary uppercase tracking-wider">{s.label}</p>
              <p className="font-[family-name:var(--font-anton)] text-[26px] md:text-[32px] mt-1">{s.val}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
