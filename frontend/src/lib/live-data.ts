/**
 * Server-side fetch of live FIFA 2026 WC match data.
 *
 * Pulls from football-data.org, joins with our pre-WC deterministic predictions,
 * computes outcome-correct / exact-score / within-1-goal / upset for finished matches.
 *
 * Called from /api/live-matches (the route) AND from /live (the page) via Next.js
 * ISR — the same logic is shared so we don't pay football-data.org twice per cache window.
 */

import { DETERMINISTIC_DATA } from "@/lib/deterministic-data";

/* ──── types ──── */

export interface TeamInfo {
  name: string | null;
  tla: string;
  flag: string | null;
}

export interface Prediction {
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  score: { home: number; away: number };
  predicted_outcome: "home" | "draw" | "away";
  source: string;
}

export interface Actual {
  score: { home: number; away: number };
  outcome: "home" | "draw" | "away";
  is_live?: boolean;          // true while the match is IN_PLAY / PAUSED (score is current, not final)
  outcome_correct?: boolean;
  exact_score?: boolean;
  within_1_goal?: boolean;
  is_upset?: boolean;
  match_score?: number;       // 0-100 from per-match Brier (only set when match is FINISHED)
  goal_diff_error?: number;   // |predicted goal-diff - actual goal-diff| (only when FINISHED)
}

export interface Match {
  id: number;
  kickoff_utc: string;
  status: string;
  matchday: number | null;
  stage: string;
  group: string | null;
  home: TeamInfo;
  away: TeamInfo;
  prediction: Prediction | null;
  actual: Actual | null;
}

export interface Stats {
  n_played: number;
  n_with_predictions?: number;
  n_correct_outcome?: number;
  outcome_accuracy?: number;
  n_exact_score?: number;
  exact_score_pct?: number;
  n_within_1_goal?: number;
  within_1_goal_pct?: number;
  n_upsets?: number;
  biggest_upset?: { match_id: number; fav_team: string; fav_prob: number; winner: string } | null;
  avg_confidence_score?: number;  // 0-100, mean of per-match match_score
  avg_goal_diff_error?: number;   // mean |predicted - actual| goal diff
}

export interface LiveData {
  last_updated: string;
  tournament_phase: string;
  stats: Stats;
  matches: Match[];
}

/* ──── name + flag mappings ──── */

const TEAM_NAME_MAP: Record<string, string> = {
  "Czechia": "Czech Republic",
  "Bosnia-Herzegovina": "Bosnia and Herzegovina",
  "Korea Republic": "South Korea",
  "Cape Verde Islands": "Cape Verde",
  "Cabo Verde": "Cape Verde",
  "Congo DR": "DR Congo",
  "Côte d'Ivoire": "Ivory Coast",
  "USA": "United States",
};

const TLA_TO_FLAG: Record<string, string | null> = {
  ALG: "dz", ARG: "ar", AUS: "au", AUT: "at", BEL: "be",
  BIH: "ba", BRA: "br", CAN: "ca", CIV: "ci", COL: "co",
  CPV: "cv", CRO: "hr", CUW: "cw", CZE: "cz",
  ECU: "ec", EGY: "eg", ENG: "gb-eng", ESP: "es",
  FRA: "fr", GER: "de", GHA: "gh", HAI: "ht", IRN: "ir",
  IRQ: "iq", JOR: "jo", JPN: "jp", KOR: "kr",
  MAR: "ma", MEX: "mx", NED: "nl", NOR: "no", NZL: "nz",
  PAN: "pa", PAR: "py", POR: "pt", QAT: "qa", RSA: "za",
  SCO: "gb-sct", SEN: "sn", SUI: "ch", SWE: "se",
  TUN: "tn", TUR: "tr", URU: "uy", USA: "us", UZB: "uz",
  COD: "cd", KSA: "sa",
};

/* ──── prediction lookup from deterministic-data ──── */

interface PredictionLookup {
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  pred_home_goals: number;
  pred_away_goals: number;
  source: string;
}

function buildPredictionMap(): Map<string, PredictionLookup> {
  const map = new Map<string, PredictionLookup>();
  const det = DETERMINISTIC_DATA as {
    groups?: Array<{ matches?: Array<{ home_team: string; away_team: string; prob_home: number; prob_draw: number; prob_away: number; home_goals: number; away_goals: number; }> }>;
    knockout?: Record<string, Array<{ team1: string; team2: string; prob_team1: number; prob_draw: number; prob_team2: number; team1_goals: number; team2_goals: number; }>>;
  };

  for (const g of det.groups ?? []) {
    for (const m of g.matches ?? []) {
      map.set(`${m.home_team}||${m.away_team}`, {
        prob_home: m.prob_home,
        prob_draw: m.prob_draw,
        prob_away: m.prob_away,
        pred_home_goals: m.home_goals,
        pred_away_goals: m.away_goals,
        source: "pre_wc_deterministic",
      });
    }
  }
  for (const [round, matches] of Object.entries(det.knockout ?? {})) {
    for (const m of matches) {
      map.set(`${m.team1}||${m.team2}`, {
        prob_home: m.prob_team1,
        prob_draw: m.prob_draw,
        prob_away: m.prob_team2,
        pred_home_goals: m.team1_goals,
        pred_away_goals: m.team2_goals,
        source: `pre_wc_deterministic_${round}`,
      });
    }
  }
  return map;
}

function predictedOutcome(p: PredictionLookup): "home" | "draw" | "away" {
  if (p.prob_home >= p.prob_draw && p.prob_home >= p.prob_away) return "home";
  if (p.prob_away >= p.prob_draw) return "away";
  return "draw";
}

function normalizeTeam(name: string): string {
  return TEAM_NAME_MAP[name] ?? name;
}

function stageLabel(stage: string, group: string | null): { stage: string; group: string | null } {
  if (stage === "GROUP_STAGE") return { stage: "group", group: group?.replace("GROUP_", "") ?? null };
  const m: Record<string, string> = {
    LAST_32: "r32", LAST_16: "r16", QUARTER_FINALS: "qf",
    SEMI_FINALS: "sf", THIRD_PLACE: "tp", FINAL: "final",
  };
  return { stage: m[stage] ?? stage.toLowerCase(), group: null };
}

/* ──── football-data.org types (only what we touch) ──── */

interface FDMatch {
  id: number;
  utcDate: string;
  status: string;
  matchday: number | null;
  stage: string;
  group: string | null;
  homeTeam: { name: string; tla: string | null };
  awayTeam: { name: string; tla: string | null };
  score: {
    winner: string | null;
    fullTime: { home: number | null; away: number | null };
  };
}

/* ──── main entry ──── */

export async function getLiveData(): Promise<LiveData> {
  const apiKey = process.env.FOOTBALL_DATA_API_KEY;
  if (!apiKey) {
    throw new Error("FOOTBALL_DATA_API_KEY not configured");
  }

  const res = await fetch(
    "https://api.football-data.org/v4/competitions/WC/matches?season=2026",
    {
      headers: { "X-Auth-Token": apiKey },
      next: { revalidate: 60 }, // 60 seconds — football-data.org free tier = 10 req/min, plenty of headroom
    }
  );

  if (!res.ok) {
    throw new Error(`football-data.org returned ${res.status}`);
  }

  const payload = (await res.json()) as { matches: FDMatch[] };
  const predMap = buildPredictionMap();

  const matches: Match[] = payload.matches.map((fdm) => {
    const homeName = normalizeTeam(fdm.homeTeam.name);
    const awayName = normalizeTeam(fdm.awayTeam.name);
    const homeTla = fdm.homeTeam.tla ?? "";
    const awayTla = fdm.awayTeam.tla ?? "";
    const stageInfo = stageLabel(fdm.stage, fdm.group);

    const record: Match = {
      id: fdm.id,
      kickoff_utc: fdm.utcDate,
      status: fdm.status,
      matchday: fdm.matchday,
      stage: stageInfo.stage,
      group: stageInfo.group,
      home: { name: homeName, tla: homeTla, flag: TLA_TO_FLAG[homeTla] ?? null },
      away: { name: awayName, tla: awayTla, flag: TLA_TO_FLAG[awayTla] ?? null },
      prediction: null,
      actual: null,
    };

    const pred = predMap.get(`${homeName}||${awayName}`);
    if (pred) {
      record.prediction = {
        prob_home: pred.prob_home,
        prob_draw: pred.prob_draw,
        prob_away: pred.prob_away,
        score: { home: pred.pred_home_goals, away: pred.pred_away_goals },
        predicted_outcome: predictedOutcome(pred),
        source: pred.source,
      };
    }

    // Use fullTime for finished matches, fall back to halfTime / regularTime for in-progress.
    // football-data.org sets fullTime to the *current* score while a match is in progress.
    const ft = fdm.score?.fullTime;
    const isFinal = fdm.status === "FINISHED";
    const isLive = fdm.status === "IN_PLAY" || fdm.status === "PAUSED" || fdm.status === "LIVE";

    if ((isFinal || isLive) && ft && ft.home != null && ft.away != null) {
      // Derive outcome from current score (for live) or fullTime winner field (for final)
      let outcome: "home" | "draw" | "away";
      if (isFinal && fdm.score?.winner) {
        outcome =
          fdm.score.winner === "HOME_TEAM" ? "home" :
          fdm.score.winner === "AWAY_TEAM" ? "away" : "draw";
      } else {
        outcome =
          ft.home > ft.away ? "home" :
          ft.away > ft.home ? "away" : "draw";
      }

      const actual: Actual = {
        score: { home: ft.home, away: ft.away },
        outcome,
        is_live: isLive,
      };

      // Only compute accuracy badges for matches that are actually over.
      if (isFinal && record.prediction) {
        const p = record.prediction;
        actual.outcome_correct = p.predicted_outcome === outcome;
        actual.exact_score = p.score.home === ft.home && p.score.away === ft.away;
        actual.within_1_goal =
          Math.abs(p.score.home - ft.home) <= 1 && Math.abs(p.score.away - ft.away) <= 1;
        // Upset = the model gave one side >=70% chance to win, and that side did NOT win.
        // Both outright losses AND draws count — a heavy favorite dropping points is
        // surprising enough to flag (e.g. Spain 85.6% held to 0-0 by Cape Verde).
        // 70% threshold is the "this team should win" line; below that, draws are
        // close enough to be expected.
        const favProb = Math.max(p.prob_home, p.prob_away);
        if (favProb >= 0.70) {
          const favOutcome = p.prob_home >= p.prob_away ? "home" : "away";
          actual.is_upset = favOutcome !== outcome;
        } else {
          actual.is_upset = false;
        }
        // Per-match Brier-derived score (0-100; higher is better)
        // brier = sum( (p_i - actual_i)^2 ), range 0-2 for 3-class. score = (1 - brier/2) * 100
        const aH = outcome === "home" ? 1 : 0;
        const aD = outcome === "draw" ? 1 : 0;
        const aA = outcome === "away" ? 1 : 0;
        const brier =
          (p.prob_home - aH) ** 2 + (p.prob_draw - aD) ** 2 + (p.prob_away - aA) ** 2;
        actual.match_score = Math.round(Math.max(0, (1 - brier / 2) * 100));
        // Goal-diff error
        const predDiff = p.score.home - p.score.away;
        const actDiff = ft.home - ft.away;
        actual.goal_diff_error = Math.abs(predDiff - actDiff);
      }
      record.actual = actual;
    }

    return record;
  });

  matches.sort((a, b) => a.kickoff_utc.localeCompare(b.kickoff_utc));

  // Only count actually-finished matches in the stats — exclude live (in-play / paused).
  const finished = matches.filter((m) => m.actual && !m.actual.is_live && m.prediction);
  const n = finished.length;
  let stats: Stats = { n_played: n };
  if (n > 0) {
    const nCorrect = finished.filter((m) => m.actual!.outcome_correct).length;
    const nExact = finished.filter((m) => m.actual!.exact_score).length;
    const nWithin1 = finished.filter((m) => m.actual!.within_1_goal).length;
    const nUpsets = finished.filter((m) => m.actual!.is_upset).length;

    let biggestUpset: Stats["biggest_upset"] = null;
    for (const m of finished) {
      if (!m.actual!.is_upset) continue;
      const p = m.prediction!;
      const favProb = Math.max(p.prob_home, p.prob_away);
      const favTeam = p.prob_home >= p.prob_away ? m.home.name : m.away.name;
      if (!biggestUpset || favProb > biggestUpset.fav_prob) {
        biggestUpset = {
          match_id: m.id,
          fav_team: favTeam ?? "Unknown",
          fav_prob: favProb,
          winner: (m.actual!.outcome === "home" ? m.home.name : m.away.name) ?? "Unknown",
        };
      }
    }

    const totalScore = finished.reduce((acc, m) => acc + (m.actual!.match_score ?? 0), 0);
    const totalGdErr = finished.reduce((acc, m) => acc + (m.actual!.goal_diff_error ?? 0), 0);

    stats = {
      n_played: n,
      n_with_predictions: n,
      n_correct_outcome: nCorrect,
      outcome_accuracy: nCorrect / n,
      n_exact_score: nExact,
      exact_score_pct: nExact / n,
      n_within_1_goal: nWithin1,
      within_1_goal_pct: nWithin1 / n,
      n_upsets: nUpsets,
      biggest_upset: biggestUpset,
      avg_confidence_score: Math.round(totalScore / n),
      avg_goal_diff_error: Math.round((totalGdErr / n) * 10) / 10, // 1 decimal
    };
  }

  return {
    last_updated: new Date().toISOString(),
    tournament_phase: matches.some((m) => m.stage === "group" && m.status !== "FINISHED")
      ? "group_stage"
      : "knockout",
    stats,
    matches,
  };
}
