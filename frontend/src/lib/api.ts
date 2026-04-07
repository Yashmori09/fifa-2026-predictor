const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function predictMatch(homeTeam: string, awayTeam: string) {
  const res = await fetch(`${API_BASE}/predict/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function simulateTournament(iterations: number = 10000) {
  const res = await fetch(`${API_BASE}/simulate/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iterations }),
  });
  if (!res.ok) throw new Error("Simulation failed");
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
