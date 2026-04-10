"use client";

import { useState, useMemo } from "react";
import { TEAMS } from "@/lib/teams";
import squadData from "@/data/squad_players.json";
import Tip from "@/components/Tip";

/* ──────────────────────── types ──────────────────────── */

interface Player {
  name: string;
  pos: string;
  age: number;
  caps: number;
  goals: number;
  ovr: number;
  pot: number;
  // outfield face stats
  pac?: number;
  sho?: number;
  pas?: number;
  dri?: number;
  defe?: number;
  phy?: number;
  // GK face stats
  div?: number;
  han?: number;
  kic?: number;
  gkp?: number;
  ref?: number;
  rep: number;
  val: number;
  club: string;
}

interface TeamStats {
  name: string;
  code: string;
  group: string;
  overall: number;
  gk: number;
  def: number;
  mid: number;
  fwd: number;
  top3: number;
  avgAge: number;
  players: Player[];
}

/* ──────────────── team stats from data ──────────────── */

const POS_ORDER: Record<string, number> = { GK: 0, DF: 1, MF: 2, FW: 3 };

const TEAM_STATS: TeamStats[] = TEAMS.map((t) => {
  const players: Player[] =
    (squadData as Record<string, Player[]>)[t.name] ?? [];
  const byPos = (pos: string) =>
    players.filter((p) => p.pos === pos).map((p) => p.ovr);
  const avg = (arr: number[]) =>
    arr.length ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : 0;

  const gkRatings = byPos("GK");
  const defRatings = byPos("DF");
  const midRatings = byPos("MF");
  const fwdRatings = byPos("FW");

  const sorted = [...players].sort((a, b) => b.ovr - a.ovr);
  const top3 = avg(sorted.slice(0, 3).map((p) => p.ovr));
  const allOvr = players.map((p) => p.ovr);
  const avgAge = avg(players.map((p) => p.age));

  return {
    name: t.name,
    code: t.code,
    group: t.group,
    overall: avg(allOvr),
    gk: avg(gkRatings),
    def: avg(defRatings),
    mid: avg(midRatings),
    fwd: avg(fwdRatings),
    top3,
    avgAge,
    players: players.sort(
      (a, b) => (POS_ORDER[a.pos] ?? 9) - (POS_ORDER[b.pos] ?? 9) || b.ovr - a.ovr
    ),
  };
}).sort((a, b) => b.overall - a.overall);

/* ──────────────── position badge colors ──────────────── */

const POS_COLORS: Record<string, string> = {
  FW: "bg-pink text-white",
  MF: "bg-green-500 text-white",
  DF: "bg-cyan text-white",
  GK: "bg-purple text-white",
};

function fmtValue(val: number): string {
  if (val >= 1_000_000) return `€${(val / 1_000_000).toFixed(1)}M`;
  if (val >= 1_000) return `€${(val / 1_000).toFixed(0)}K`;
  return val > 0 ? `€${val}` : "—";
}

function statColor(val: number): string {
  if (val >= 85) return "text-green-400";
  if (val >= 75) return "text-lime-400";
  if (val >= 65) return "text-amber-400";
  return "text-red-400";
}

const REP_STARS = ["", "★", "★★", "★★★", "★★★★", "★★★★★"];

/* ──────────────────────── page ──────────────────────── */

export default function SquadsPage() {
  const [selectedTeam, setSelectedTeam] = useState(TEAM_STATS[0].name);

  const team = useMemo(
    () => TEAM_STATS.find((t) => t.name === selectedTeam) ?? TEAM_STATS[0],
    [selectedTeam]
  );

  const statCards = [
    { label: "GK", value: team.gk, color: "text-purple" },
    { label: "DEF", value: team.def, color: "text-cyan" },
    { label: "MID", value: team.mid, color: "text-green-500" },
    { label: "FWD", value: team.fwd, color: "text-pink" },
    { label: "Top 3", value: team.top3, color: "text-amber-400" },
    { label: "Avg Age", value: team.avgAge, color: "text-secondary" },
  ];

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center px-4 md:px-12 lg:px-20 py-8 md:py-10 gap-2.5">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          SQUAD DATA
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] lg:text-[64px] leading-none">
          SQUAD EXPLORER
        </h1>
        <p className="text-secondary text-xs md:text-sm max-w-[600px]">
          1,054 players across 48 World Cup nations. EA FC ratings, positional
          depth, and squad profiles.
        </p>
        <span className="text-secondary text-[10px] md:text-[11px] bg-[#1A1A1A] border border-border rounded px-3 py-1.5 w-fit mt-1">
          Squads reflect the most recent international call-ups per team. Final 2026 World Cup selections may differ.
        </span>
      </section>

      {/* Main content */}
      <section className="flex flex-col lg:flex-row px-4 md:px-12 lg:px-20 py-6 md:py-8 gap-6 md:gap-8 border-t border-border min-h-0 lg:min-h-[calc(100vh-260px)]">
        {/* Team selector — horizontal scroll on mobile, sidebar on desktop */}
        <div className="flex flex-col lg:w-[260px] shrink-0">
          <h3 className="font-[family-name:var(--font-anton)] text-sm tracking-wide mb-3">
            SELECT TEAM
          </h3>
          <div className="flex lg:flex-col gap-1.5 lg:gap-1 overflow-x-auto lg:overflow-x-visible lg:overflow-y-auto lg:max-h-[calc(100vh-340px)] pb-2 lg:pb-0 lg:pr-2">
            {TEAM_STATS.map((t) => (
              <button
                key={t.name}
                onClick={() => setSelectedTeam(t.name)}
                className={`flex items-center justify-between px-3 py-2.5 rounded-lg text-left transition-colors shrink-0 ${
                  selectedTeam === t.name
                    ? "bg-purple text-white"
                    : "hover:bg-[#1A1A1A] text-secondary"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <span className={`fi fi-${t.code} text-base`} />
                  <span className="text-[13px] font-medium">{t.name}</span>
                </div>
                <span
                  className={`font-mono text-xs ${
                    selectedTeam === t.name ? "text-white/80" : "text-secondary"
                  }`}
                >
                  {t.overall}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Right — team profile */}
        <div className="flex-1 flex flex-col gap-6">
          {/* Team header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 md:gap-4">
              <span className={`fi fi-${team.code} text-2xl md:text-4xl`} />
              <div>
                <h2 className="font-[family-name:var(--font-anton)] text-[24px] md:text-[36px] leading-none">
                  {team.name}
                </h2>
                <span className="text-secondary text-[13px]">
                  Group {team.group} &middot; {team.players.length} players
                </span>
              </div>
            </div>
            <div className="flex flex-col items-end">
              <span className="font-[family-name:var(--font-anton)] text-[28px] md:text-[40px] text-purple leading-none">
                {team.overall}
              </span>
              <span className="text-secondary text-xs">AVG OVERALL</span>
            </div>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2 md:gap-3">
            {statCards.map((s) => (
              <div
                key={s.label}
                className="flex flex-col items-center gap-1 bg-[#111111] border border-[#1A1A1A] rounded-lg py-4"
              >
                <span
                  className={`font-[family-name:var(--font-anton)] text-[28px] ${s.color}`}
                >
                  {s.value}
                </span>
                <span className="text-secondary text-[11px] font-medium">
                  {s.label}
                </span>
              </div>
            ))}
          </div>

          {/* Outfield players table */}
          <div className="flex flex-col overflow-x-auto">
            <table className="w-full min-w-[1020px] text-[13px]">
              <thead>
                <tr className="text-[11px] text-secondary font-semibold tracking-wider border-b border-border">
                  <th className="text-left px-3 py-2 w-[160px]">PLAYER</th>
                  <th className="text-center px-1 py-2 w-[40px]"><Tip term="Pos">POS</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]">AGE</th>
                  <th className="text-center px-1 py-2 w-[42px]"><Tip term="OVR">OVR</Tip></th>
                  <th className="text-center px-1 py-2 w-[42px]"><Tip term="POT">POT</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="PAC">PAC</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="SHO">SHO</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="PAS">PAS</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="DRI">DRI</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="DEF">DEF</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="PHY">PHY</Tip></th>
                  <th className="text-center px-1 py-2 w-[44px]"><Tip term="caps">CAPS</Tip></th>
                  <th className="text-center px-1 py-2 w-[44px]">GOALS</th>
                  <th className="text-center px-1 py-2 w-[52px]"><Tip term="REP">REP</Tip></th>
                  <th className="text-center px-1 py-2 w-[72px]"><Tip term="VAL">VALUE</Tip></th>
                  <th className="text-left px-3 py-2">CLUB</th>
                </tr>
              </thead>
              <tbody>
                {team.players
                  .filter((p) => p.pos !== "GK")
                  .map((p, i) => (
                    <tr
                      key={`${p.name}-${i}`}
                      className="border-b border-[#1A1A1A] hover:bg-[#111111] transition-colors"
                    >
                      <td className="px-3 py-2 font-medium truncate max-w-[160px]">
                        {p.name}
                      </td>
                      <td className="text-center px-1 py-2">
                        <span
                          className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            POS_COLORS[p.pos] ?? "bg-[#2A2A2A] text-secondary"
                          }`}
                        >
                          {p.pos}
                        </span>
                      </td>
                      <td className="text-center px-1 py-2 text-secondary">{p.age}</td>
                      <td className="text-center px-1 py-2 font-semibold text-purple">{p.ovr}</td>
                      <td className="text-center px-1 py-2 font-semibold text-cyan">{p.pot}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.pac ?? 0)}`}>{p.pac || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.sho ?? 0)}`}>{p.sho || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.pas ?? 0)}`}>{p.pas || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.dri ?? 0)}`}>{p.dri || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.defe ?? 0)}`}>{p.defe || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.phy ?? 0)}`}>{p.phy || "—"}</td>
                      <td className="text-center px-1 py-2 text-secondary">{p.caps}</td>
                      <td className="text-center px-1 py-2 text-secondary">{p.goals}</td>
                      <td className="text-center px-1 py-2 text-amber-400 text-[10px]">{REP_STARS[p.rep] ?? "★"}</td>
                      <td className="text-center px-1 py-2 font-mono text-xs text-secondary">{fmtValue(p.val)}</td>
                      <td className="px-3 py-2 text-[#A1A1AA] truncate max-w-[140px]">{p.club}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Goalkeepers table */}
          <div className="flex flex-col overflow-x-auto mt-2">
            <h4 className="font-[family-name:var(--font-anton)] text-sm tracking-wide mb-2 text-purple">
              GOALKEEPERS
            </h4>
            <table className="w-full min-w-[920px] text-[13px]">
              <thead>
                <tr className="text-[11px] text-secondary font-semibold tracking-wider border-b border-border">
                  <th className="text-left px-3 py-2 w-[160px]">PLAYER</th>
                  <th className="text-center px-1 py-2 w-[36px]">AGE</th>
                  <th className="text-center px-1 py-2 w-[42px]"><Tip term="OVR">OVR</Tip></th>
                  <th className="text-center px-1 py-2 w-[42px]"><Tip term="POT">POT</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="DIV">DIV</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="HAN">HAN</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="KIC">KIC</Tip></th>
                  <th className="text-center px-1 py-2 w-[36px]" title="Goalkeeper Positioning">POS</th>
                  <th className="text-center px-1 py-2 w-[36px]"><Tip term="REF">REF</Tip></th>
                  <th className="text-center px-1 py-2 w-[44px]"><Tip term="caps">CAPS</Tip></th>
                  <th className="text-center px-1 py-2 w-[44px]">GOALS</th>
                  <th className="text-center px-1 py-2 w-[52px]"><Tip term="REP">REP</Tip></th>
                  <th className="text-center px-1 py-2 w-[72px]"><Tip term="VAL">VALUE</Tip></th>
                  <th className="text-left px-3 py-2">CLUB</th>
                </tr>
              </thead>
              <tbody>
                {team.players
                  .filter((p) => p.pos === "GK")
                  .map((p, i) => (
                    <tr
                      key={`gk-${p.name}-${i}`}
                      className="border-b border-[#1A1A1A] hover:bg-[#111111] transition-colors"
                    >
                      <td className="px-3 py-2 font-medium truncate max-w-[160px]">{p.name}</td>
                      <td className="text-center px-1 py-2 text-secondary">{p.age}</td>
                      <td className="text-center px-1 py-2 font-semibold text-purple">{p.ovr}</td>
                      <td className="text-center px-1 py-2 font-semibold text-cyan">{p.pot}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.div ?? 0)}`}>{p.div || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.han ?? 0)}`}>{p.han || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.kic ?? 0)}`}>{p.kic || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.gkp ?? 0)}`}>{p.gkp || "—"}</td>
                      <td className={`text-center px-1 py-2 font-mono text-xs ${statColor(p.ref ?? 0)}`}>{p.ref || "—"}</td>
                      <td className="text-center px-1 py-2 text-secondary">{p.caps}</td>
                      <td className="text-center px-1 py-2 text-secondary">{p.goals}</td>
                      <td className="text-center px-1 py-2 text-amber-400 text-[10px]">{REP_STARS[p.rep] ?? "★"}</td>
                      <td className="text-center px-1 py-2 font-mono text-xs text-secondary">{fmtValue(p.val)}</td>
                      <td className="px-3 py-2 text-[#A1A1AA] truncate max-w-[140px]">{p.club}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

    </div>
  );
}
