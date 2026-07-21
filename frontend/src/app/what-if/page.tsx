import { promises as fs } from "fs";
import path from "path";
import WhatIfClient from "./WhatIfClient";

export const dynamic = "force-static";

export interface WhatIfMatch {
  id: string;
  stage: "r32" | "r16" | "qf" | "sf" | "final";
  team1: string;
  team2: string;
  team1_goals: number;
  team2_goals: number;
  winner: string;
  parent1: string | null;
  parent2: string | null;
}

export interface WhatIfData {
  generated_at: string;
  tree: {
    r32: WhatIfMatch[];
    r16: WhatIfMatch[];
    qf: WhatIfMatch[];
    sf: WhatIfMatch[];
    final: WhatIfMatch[];
  };
  pair_win: Record<string, number>;
  r32_teams: string[];
  champion: string;
}

export default async function WhatIfPage() {
  const file = path.join(process.cwd(), "public", "whatif-data.json");
  const raw = await fs.readFile(file, "utf-8");
  const data: WhatIfData = JSON.parse(raw);
  return <WhatIfClient data={data} />;
}
