"use client";

import { useState, useEffect, useRef } from "react";
import { TEAMS } from "@/lib/teams";

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

const ELO_MAP: Record<string, number> = {
  Spain: 2270, Argentina: 2227, France: 2211, England: 2149, Colombia: 2096,
  Brazil: 2086, Portugal: 2054, Netherlands: 2044, Japan: 2041, Germany: 2034,
  Ecuador: 2032, Croatia: 2029, Norway: 2023, Uruguay: 2015, Turkey: 2015,
  Switzerland: 2007, Morocco: 1989, Mexico: 1987, Senegal: 1982, Paraguay: 1959,
  Belgium: 1927, Australia: 1924, Austria: 1921, Canada: 1917, "South Korea": 1888,
  Iran: 1887, Panama: 1886, Scotland: 1872, Algeria: 1869, Uzbekistan: 1864,
  "United States": 1856, Sweden: 1855, Jordan: 1837, Egypt: 1826, "DR Congo": 1823,
  "Ivory Coast": 1814, "Czech Republic": 1812, "New Zealand": 1780, Tunisia: 1765,
  Iraq: 1746, "Saudi Arabia": 1721, "Cape Verde": 1699, "Bosnia and Herzegovina": 1686,
  Ghana: 1670, Haiti: 1664, "Curaçao": 1653, "South Africa": 1642, Qatar: 1585,
};

/* ── Animated counter hook ── */
function useCountUp(target: number, duration: number, active: boolean) {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!active) { setValue(0); return; }
    const start = performance.now();
    const animate = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration, active]);

  return value;
}

export default function PredictPage() {
  const [homeTeam, setHomeTeam] = useState("Spain");
  const [awayTeam, setAwayTeam] = useState("Argentina");
  const [predicting, setPredicting] = useState(false);
  const [showHomeSelector, setShowHomeSelector] = useState(false);
  const [showAwaySelector, setShowAwaySelector] = useState(false);
  const [result, setResult] = useState<{
    home_win: number;
    draw: number;
    away_win: number;
  } | null>(null);
  const [animating, setAnimating] = useState(false);
  // Tracks whether we just got fresh results (triggers count-up)
  const [freshResult, setFreshResult] = useState(false);

  const homeData = TEAMS.find((t) => t.name === homeTeam);
  const awayData = TEAMS.find((t) => t.name === awayTeam);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;
    setPredicting(true);
    setAnimating(true);
    setResult(null);
    setFreshResult(false);

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/predict/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult({
        home_win: data.home_win,
        draw: data.draw,
        away_win: data.away_win,
      });
      setFreshResult(true);
    } catch {
      setResult(null);
    } finally {
      setPredicting(false);
      setTimeout(() => setAnimating(false), 900);
    }
  };

  // Count-up values
  const homeCount = useCountUp(
    result ? result.home_win * 100 : 0, 800, freshResult
  );
  const drawCount = useCountUp(
    result ? result.draw * 100 : 0, 800, freshResult
  );
  const awayCount = useCountUp(
    result ? result.away_win * 100 : 0, 800, freshResult
  );

  // Use counted values when animating, final values otherwise
  const displayHome = freshResult ? homeCount : (result ? result.home_win * 100 : 0);
  const displayDraw = freshResult ? drawCount : (result ? result.draw * 100 : 0);
  const displayAway = freshResult ? awayCount : (result ? result.away_win * 100 : 0);

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center h-[140px] px-20 gap-2.5">
        <p className="text-pink text-[11px] font-semibold tracking-[3px]">
          AI MATCH PREDICTION
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[48px] leading-tight">
          HEAD TO HEAD PREDICTOR
        </h1>
        <p className="text-secondary text-sm">
          Select any two nations from all 48 World Cup teams to get instant
          AI-powered match probabilities
        </p>
      </section>

      {/* Main content */}
      <section className="flex gap-8 px-20 py-12">
        {/* Left — Team Selector */}
        <div className="flex flex-col gap-6 w-[680px]">
          {/* Teams row */}
          <div className="flex gap-4">
            {/* Home team card */}
            <div className={`flex-1 flex flex-col items-center gap-4 bg-[#111111] border-2 rounded-xl px-8 py-9 transition-all duration-300 ${
              predicting ? "border-purple shadow-[0_0_20px_rgba(168,85,247,0.3)]" : "border-purple"
            }`}>
              {homeData && (
                <span className={`fi fi-${homeData.code} text-6xl transition-transform duration-300 ${
                  predicting ? "scale-110" : ""
                }`} />
              )}
              <span className="font-[family-name:var(--font-anton)] text-[28px]">
                {homeTeam || "Select"}
              </span>
              <span className="font-mono text-[11px] text-secondary">
                {CONF_MAP[homeTeam] || "—"} · ELO {ELO_MAP[homeTeam] || "—"}
              </span>
              <div className="relative w-full">
                <button
                  onClick={() => { setShowHomeSelector(!showHomeSelector); setShowAwaySelector(false); }}
                  className="w-full h-9 bg-[#1A1A1A] rounded text-xs text-secondary hover:text-foreground transition-colors"
                >
                  Change Team
                </button>
                {showHomeSelector && (
                  <div className="absolute top-10 left-0 right-0 bg-[#1A1A1A] border border-border rounded-lg max-h-[240px] overflow-y-auto z-20">
                    {TEAMS.filter((t) => t.name !== awayTeam).map((t) => (
                      <button
                        key={t.name}
                        onClick={() => { setHomeTeam(t.name); setShowHomeSelector(false); setResult(null); setFreshResult(false); }}
                        className="flex items-center gap-2 w-full px-3 py-2 text-xs text-left hover:bg-[#222] transition-colors"
                      >
                        <span className={`fi fi-${t.code}`} />
                        {t.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* VS */}
            <div className="flex flex-col items-center justify-center gap-2 w-[60px]">
              <span className={`font-[family-name:var(--font-anton)] text-[44px] text-purple transition-transform duration-300 ${
                predicting ? "scale-125" : ""
              }`}>
                VS
              </span>
              <div className="w-0.5 h-[60px] bg-[#1A1A1A]" />
            </div>

            {/* Away team card */}
            <div className={`flex-1 flex flex-col items-center gap-4 bg-[#111111] border-2 rounded-xl px-8 py-9 transition-all duration-300 ${
              predicting ? "border-pink shadow-[0_0_20px_rgba(236,72,153,0.3)]" : "border-[#1A1A1A]"
            }`}>
              {awayData && (
                <span className={`fi fi-${awayData.code} text-6xl transition-transform duration-300 ${
                  predicting ? "scale-110" : ""
                }`} />
              )}
              <span className="font-[family-name:var(--font-anton)] text-[28px]">
                {awayTeam || "Select"}
              </span>
              <span className="font-mono text-[11px] text-secondary">
                {CONF_MAP[awayTeam] || "—"} · ELO {ELO_MAP[awayTeam] || "—"}
              </span>
              <div className="relative w-full">
                <button
                  onClick={() => { setShowAwaySelector(!showAwaySelector); setShowHomeSelector(false); }}
                  className="w-full h-9 bg-[#1A1A1A] rounded text-xs text-secondary hover:text-foreground transition-colors"
                >
                  Change Team
                </button>
                {showAwaySelector && (
                  <div className="absolute top-10 left-0 right-0 bg-[#1A1A1A] border border-border rounded-lg max-h-[240px] overflow-y-auto z-20">
                    {TEAMS.filter((t) => t.name !== homeTeam).map((t) => (
                      <button
                        key={t.name}
                        onClick={() => { setAwayTeam(t.name); setShowAwaySelector(false); setResult(null); setFreshResult(false); }}
                        className="flex items-center gap-2 w-full px-3 py-2 text-xs text-left hover:bg-[#222] transition-colors"
                      >
                        <span className={`fi fi-${t.code}`} />
                        {t.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Predict button */}
          <button
            onClick={handlePredict}
            disabled={!homeTeam || !awayTeam || homeTeam === awayTeam || predicting}
            className={`w-full h-14 rounded-md bg-gradient-to-r from-purple to-pink text-white font-[family-name:var(--font-anton)] text-lg tracking-[3px] transition-all disabled:opacity-30 ${
              predicting ? "opacity-70 animate-pulse" : "hover:opacity-90"
            }`}
          >
            {predicting ? "PREDICTING..." : "PREDICT MATCH"}
          </button>
        </div>

        {/* Right — Results Panel */}
        <div className="flex-1 flex flex-col gap-5">
          <h2 className="font-[family-name:var(--font-anton)] text-xl tracking-wide">
            PREDICTION RESULT
          </h2>

          {result ? (
            <div className={`flex flex-col gap-5 bg-[#111111] border border-[#1A1A1A] rounded-lg p-7 transition-all duration-500 ${
              freshResult ? "opacity-100 translate-y-0" : "opacity-100"
            }`}>
              <span className="text-secondary text-[13px]">
                {homeTeam} vs {awayTeam}
              </span>

              {/* Home win bar */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-sm">
                  <span>
                    <span className="text-purple font-mono text-xs">01</span>{" "}
                    <span className="font-semibold">{homeTeam} Win</span>
                  </span>
                  <span className="font-mono text-cyan tabular-nums">
                    {displayHome.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 bg-[#1A1A1A] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple to-cyan rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${freshResult ? displayHome : result.home_win * 100}%` }}
                  />
                </div>
              </div>

              {/* Draw bar */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-sm">
                  <span className="font-semibold">Draw</span>
                  <span className="font-mono text-secondary tabular-nums">
                    {displayDraw.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 bg-[#1A1A1A] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-secondary rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${freshResult ? displayDraw : result.draw * 100}%` }}
                  />
                </div>
              </div>

              {/* Away win bar */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-sm">
                  <span>
                    <span className="text-pink font-mono text-xs">02</span>{" "}
                    <span className="font-semibold">{awayTeam} Win</span>
                  </span>
                  <span className="font-mono text-pink tabular-nums">
                    {displayAway.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 bg-[#1A1A1A] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-pink to-purple rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${freshResult ? displayAway : result.away_win * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ) : predicting ? (
            <div className="flex-1 flex flex-col items-center justify-center bg-[#111111] border border-[#1A1A1A] rounded-lg p-7 min-h-[200px] gap-3">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 rounded-full bg-purple animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 rounded-full bg-cyan animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 rounded-full bg-pink animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <span className="text-secondary text-sm">
                Running prediction model...
              </span>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-[#111111] border border-[#1A1A1A] rounded-lg p-7 min-h-[200px]">
              <span className="text-secondary text-sm">
                Select two teams and click Predict Match
              </span>
            </div>
          )}

        </div>
      </section>
    </div>
  );
}
