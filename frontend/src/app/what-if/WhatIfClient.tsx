"use client";

import { useState, useMemo } from "react";
import { TEAMS } from "@/lib/teams";
import type { WhatIfData, WhatIfMatch } from "./page";

function teamCode(name: string): string {
  return TEAMS.find((t) => t.name === name)?.code ?? "xx";
}

const STAGE_LABEL: Record<string, string> = {
  r32: "R32",
  r16: "R16",
  qf: "QF",
  sf: "SF",
  final: "Final",
};

const STAGE_ORDER: Array<"r32" | "r16" | "qf" | "sf" | "final"> = ["r32", "r16", "qf", "sf", "final"];

/* ─────────── cascade computation ─────────── */

interface ComputedMatch {
  id: string;
  stage: string;
  team1: string;
  team2: string;
  winner: string;
  isNewPairing: boolean;   // did an ancestor flip create this pairing?
  isFlipped: boolean;      // did the user flip this match directly?
  isDivergent: boolean;    // is the winner different from the actual tournament?
}

function computeBracket(
  tree: WhatIfData["tree"],
  pairWin: Record<string, number>,
  flips: Set<string>
): Map<string, ComputedMatch> {
  const flat: WhatIfMatch[] = [
    ...tree.r32, ...tree.r16, ...tree.qf, ...tree.sf, ...tree.final,
  ];
  const byId = new Map<string, WhatIfMatch>();
  flat.forEach((m) => byId.set(m.id, m));

  const computed = new Map<string, ComputedMatch>();

  function compute(id: string): ComputedMatch {
    if (computed.has(id)) return computed.get(id)!;
    const match = byId.get(id)!;

    let team1 = match.team1;
    let team2 = match.team2;
    let isNewPairing = false;

    if (match.parent1) {
      const p1 = compute(match.parent1);
      team1 = p1.winner;
      if (team1 !== match.team1) isNewPairing = true;
    }
    if (match.parent2) {
      const p2 = compute(match.parent2);
      team2 = p2.winner;
      if (team2 !== match.team2) isNewPairing = true;
    }

    // Natural winner: actual result if unchanged, model argmax if new pairing
    let naturalWinner: string;
    if (!isNewPairing) {
      naturalWinner = match.winner;
    } else {
      const key = `${team1}||${team2}`;
      const p = pairWin[key] ?? 0.5;
      naturalWinner = p >= 0.5 ? team1 : team2;
    }

    const isFlipped = flips.has(id);
    const winner = isFlipped
      ? (naturalWinner === team1 ? team2 : team1)
      : naturalWinner;

    const isDivergent = winner !== match.winner;

    const result: ComputedMatch = {
      id, stage: match.stage,
      team1, team2, winner,
      isNewPairing, isFlipped, isDivergent,
    };
    computed.set(id, result);
    return result;
  }

  flat.forEach((m) => compute(m.id));
  return computed;
}

/* ─────────── match card ─────────── */

function MatchCard({
  match,
  computed,
  onFlip,
  actualWinner,
}: {
  match: WhatIfMatch;
  computed: ComputedMatch;
  onFlip: () => void;
  actualWinner: string;
}) {
  const winnerIsT1 = computed.winner === computed.team1;

  return (
    <div
      className={`relative bg-surface border rounded-lg p-2 transition-all ${
        computed.isDivergent
          ? "border-pink/50"
          : computed.isNewPairing
          ? "border-cyan/30"
          : "border-border"
      }`}
    >
      {/* Team 1 row */}
      <div
        className={`flex items-center gap-2 py-1 px-2 rounded ${
          winnerIsT1 ? "bg-purple/10" : ""
        }`}
      >
        <span className={`fi fi-${teamCode(computed.team1)} text-sm`} />
        <span
          className={`text-[11px] flex-1 truncate ${
            winnerIsT1 ? "text-foreground font-semibold" : "text-secondary"
          }`}
        >
          {computed.team1}
        </span>
        {winnerIsT1 && !computed.isNewPairing && (
          <span className="font-mono text-[10px] text-secondary">
            {computed.team1 === match.team1 ? match.team1_goals : "—"}
          </span>
        )}
      </div>
      {/* Team 2 row */}
      <div
        className={`flex items-center gap-2 py-1 px-2 rounded ${
          !winnerIsT1 ? "bg-purple/10" : ""
        }`}
      >
        <span className={`fi fi-${teamCode(computed.team2)} text-sm`} />
        <span
          className={`text-[11px] flex-1 truncate ${
            !winnerIsT1 ? "text-foreground font-semibold" : "text-secondary"
          }`}
        >
          {computed.team2}
        </span>
        {!winnerIsT1 && !computed.isNewPairing && (
          <span className="font-mono text-[10px] text-secondary">
            {computed.team2 === match.team2 ? match.team2_goals : "—"}
          </span>
        )}
      </div>

      {/* Flip button */}
      <button
        onClick={onFlip}
        className={`absolute -right-2 -top-2 w-5 h-5 rounded-full text-[10px] flex items-center justify-center transition-all ${
          computed.isFlipped
            ? "bg-pink text-white"
            : "bg-border text-secondary hover:bg-purple hover:text-white"
        }`}
        title={computed.isFlipped ? "Un-flip this match" : "Flip this match"}
      >
        ⇄
      </button>

      {/* Divergence/new indicator */}
      {computed.isNewPairing && (
        <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 bg-cyan text-background text-[8px] px-1.5 py-0.5 rounded font-mono">
          MODEL
        </div>
      )}
    </div>
  );
}

/* ─────────── main ─────────── */

export default function WhatIfClient({ data }: { data: WhatIfData }) {
  const [flips, setFlips] = useState<Set<string>>(new Set());

  const computed = useMemo(
    () => computeBracket(data.tree, data.pair_win, flips),
    [data.tree, data.pair_win, flips]
  );

  const toggleFlip = (id: string) => {
    setFlips((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const reset = () => setFlips(new Set());

  const newChampion = computed.get("final_0")?.winner ?? data.champion;
  const championChanged = newChampion !== data.champion;

  // Divergence stats
  const totalDivergent = Array.from(computed.values()).filter((c) => c.isDivergent).length;

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center min-h-[200px] md:min-h-[280px] px-4 md:px-12 lg:px-20 py-8 md:py-0 gap-3 md:gap-4">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          WHAT IF · ALTERNATIVE TIMELINES
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[40px] md:text-[56px] lg:text-[72px] leading-[1] max-w-[860px]">
          REWRITE
          <br />
          HISTORY
        </h1>
        <p className="text-secondary text-[13px] md:text-[15px] max-w-[820px]">
          Click <span className="font-mono text-cyan">⇄</span> on any knockout match to flip it.
          The rest of the tournament re-cascades using our Phase 3 hybrid model. New pairings that
          never happened get <span className="text-cyan">MODEL</span> predictions.
        </p>
        <div className="w-[60px] h-1 bg-purple rounded-sm" />
      </section>

      {/* Status bar */}
      <section className="sticky top-14 md:top-16 z-40 bg-background/95 backdrop-blur border-y border-border px-4 md:px-12 lg:px-20 py-3 md:py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="flex items-center gap-3 md:gap-6 flex-wrap">
            <div className="flex flex-col">
              <span className="font-mono text-[9px] text-secondary uppercase tracking-wider">Actual champion</span>
              <div className="flex items-center gap-2">
                <span className={`fi fi-${teamCode(data.champion)} text-sm`} />
                <span className="font-[family-name:var(--font-anton)] text-[16px] md:text-[20px]">{data.champion}</span>
              </div>
            </div>
            <span className="text-secondary text-[16px]">→</span>
            <div className="flex flex-col">
              <span className="font-mono text-[9px] text-secondary uppercase tracking-wider">This timeline</span>
              <div className="flex items-center gap-2">
                <span className={`fi fi-${teamCode(newChampion)} text-sm`} />
                <span className={`font-[family-name:var(--font-anton)] text-[16px] md:text-[20px] ${
                  championChanged ? "text-pink" : "text-foreground"
                }`}>{newChampion}</span>
              </div>
            </div>
            {flips.size > 0 && (
              <div className="flex flex-col">
                <span className="font-mono text-[9px] text-secondary uppercase tracking-wider">Flipped</span>
                <span className="font-mono text-[13px] text-cyan">{flips.size} match{flips.size > 1 ? "es" : ""}</span>
              </div>
            )}
            {totalDivergent > 0 && (
              <div className="flex flex-col">
                <span className="font-mono text-[9px] text-secondary uppercase tracking-wider">Divergent</span>
                <span className="font-mono text-[13px] text-pink">{totalDivergent} match{totalDivergent > 1 ? "es" : ""}</span>
              </div>
            )}
          </div>
          <button
            onClick={reset}
            disabled={flips.size === 0}
            className={`text-[11px] font-mono px-3 py-1.5 rounded border transition-all ${
              flips.size === 0
                ? "border-border text-secondary/50 cursor-not-allowed"
                : "border-purple text-purple hover:bg-purple hover:text-white"
            }`}
          >
            RESET TIMELINE
          </button>
        </div>
      </section>

      {/* Bracket */}
      <section className="px-2 md:px-8 lg:px-16 py-6 md:py-10 overflow-x-auto">
        <div className="min-w-[900px] grid grid-cols-5 gap-3 md:gap-5">
          {STAGE_ORDER.map((stage) => (
            <div key={stage} className="flex flex-col gap-2 md:gap-3">
              <div className="text-center">
                <p className="font-[family-name:var(--font-anton)] text-[14px] md:text-[18px] tracking-wide">
                  {STAGE_LABEL[stage]}
                </p>
                <p className="font-mono text-[9px] text-secondary">{data.tree[stage].length} match{data.tree[stage].length > 1 ? "es" : ""}</p>
              </div>
              <div
                className="flex flex-col gap-2 md:gap-3 flex-1 justify-around"
                style={{ minHeight: 640 }}
              >
                {data.tree[stage].map((m) => {
                  const c = computed.get(m.id)!;
                  return (
                    <MatchCard
                      key={m.id}
                      match={m}
                      computed={c}
                      onFlip={() => toggleFlip(m.id)}
                      actualWinner={m.winner}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Legend */}
      <section className="px-4 md:px-12 lg:px-20 py-6 md:py-8 border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[18px] md:text-[22px] tracking-wide mb-3">LEGEND</h2>
        <div className="flex flex-wrap gap-4 md:gap-8 text-[12px]">
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded border border-border bg-surface" />
            <span className="text-secondary">Actual result (unflipped)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded border border-cyan/50 bg-surface" />
            <span className="text-secondary">New pairing (ancestor was flipped) — <span className="text-cyan">MODEL</span> prediction</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded border border-pink/50 bg-surface" />
            <span className="text-secondary">Divergent — winner differs from reality</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded-full bg-pink flex items-center justify-center text-white text-[9px]">⇄</span>
            <span className="text-secondary">Flipped by you</span>
          </div>
        </div>
      </section>

      {/* Explainer */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-3">
          HOW THIS WORKS
        </h2>
        <div className="text-secondary text-[13px] md:text-[14px] leading-relaxed max-w-[880px] space-y-3">
          <p>
            The bracket starts with actual WC 2026 results. When you flip a match, that match&apos;s{" "}
            loser advances instead. Every downstream match then re-cascades:
          </p>
          <ul className="list-disc list-inside space-y-1 pl-2">
            <li>If the pairing at the next match is the SAME as actual reality, we keep the real result.</li>
            <li>If the pairing is NEW (because your flip changed who advanced), the winner is chosen by our Phase 3 hybrid model — argmax of P(A beats B) at neutral venue.</li>
            <li>You can keep flipping multiple matches at any stage. Effects compound.</li>
          </ul>
          <p>
            <span className="text-foreground">Try:</span> flip the Final. Or flip a big R32 upset. Or flip
            a few Group-A R32 matches at once and watch the top of the bracket rearrange.
          </p>
        </div>
      </section>
    </div>
  );
}
