import { promises as fs } from "fs";
import path from "path";
import RetrospectiveClient from "./RetrospectiveClient";

export const dynamic = "force-static";

export interface RetroTeam {
  team: string;
  flag: string;
  predicted: {
    p_r16: number;
    p_qf: number;
    p_sf: number;
    p_final: number;
    p_win: number;
    expected_finish: string;
  };
  actual: {
    finish_stage: string;
    finish_label: string;
    matches_played: number;
    wins: number;
    draws: number;
    losses: number;
    gf: number;
    ga: number;
  };
  elimination: {
    opponent: string;
    opponent_flag: string;
    score: string;
    stage: string;
    stage_label: string;
    our_win_prob?: number;
    opp_win_prob?: number;
    draw_prob?: number;
    our_pick?: string;
  } | null;
  verdict: "overachieved" | "matched" | "underperformed";
}

export interface RetroData {
  generated_at: string;
  overall: {
    n_teams: number;
    champion: string;
    runner_up: string;
    stats: {
      n_played: number;
      outcome_accuracy: number;
      avg_confidence_score: number;
      n_upsets: number;
    };
  };
  teams: Record<string, RetroTeam>;
}

export default async function RetrospectivePage() {
  const file = path.join(process.cwd(), "public", "retrospective.json");
  const raw = await fs.readFile(file, "utf-8");
  const data: RetroData = JSON.parse(raw);
  return <RetrospectiveClient data={data} />;
}
