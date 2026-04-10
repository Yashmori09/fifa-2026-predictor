import Link from "next/link";
import Tip from "@/components/Tip";

const CHART_DATA = [
  { rank: "01", team: "Spain", code: "es", pct: 21.40, top: true },
  { rank: "02", team: "France", code: "fr", pct: 21.03, top: true },
  { rank: "03", team: "Argentina", code: "ar", pct: 10.83, top: true },
  { rank: "04", team: "England", code: "gb-eng", pct: 9.72, top: true },
  { rank: "05", team: "Germany", code: "de", pct: 9.50, top: true },
  { rank: "06", team: "Brazil", code: "br", pct: 6.79, top: false },
  { rank: "07", team: "Netherlands", code: "nl", pct: 4.88, top: false },
  { rank: "08", team: "Portugal", code: "pt", pct: 2.42, top: false },
  { rank: "09", team: "Norway", code: "no", pct: 2.26, top: false },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center min-h-[200px] md:min-h-[280px] px-4 md:px-12 lg:px-20 py-8 md:py-0 gap-3 md:gap-4">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          AI-POWERED TOURNAMENT SIMULATION
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[40px] md:text-[56px] lg:text-[72px] leading-[1] max-w-[860px]">
          WHO WINS THE
          <br />
          2026 WORLD CUP?
        </h1>
        <p className="text-secondary text-[13px] md:text-[15px] max-w-[820px]">
          10,000 <Tip term="Monte Carlo">Monte Carlo</Tip> simulations &middot; Player-rated squad analysis
        </p>
        <div className="w-[60px] h-1 bg-purple rounded-sm" />
      </section>

      {/* Championship Probability */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-2">
          <div>
            <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide">
              CHAMPIONSHIP PROBABILITY
            </h2>
            <p className="text-secondary text-[11px] mt-1">How often each team wins across 10,000 simulated tournaments</p>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          {CHART_DATA.map((row) => (
            <div key={row.rank} className="flex items-center gap-2 md:gap-4 h-10">
              <span
                className={`font-mono text-xs w-6 ${
                  row.top ? "text-purple" : "text-secondary"
                }`}
              >
                {row.rank}
              </span>
              <span className={`fi fi-${row.code} text-lg`} />
              <span
                className={`text-xs md:text-sm font-semibold w-20 md:w-28 ${
                  row.top ? "text-foreground" : "text-secondary"
                }`}
              >
                {row.team}
              </span>
              <div className="flex-1 h-7 bg-[#1A1A1A] rounded">
                <div
                  className={`h-full rounded ${
                    row.top
                      ? "bg-gradient-to-r from-purple to-cyan"
                      : "bg-[#2A2A2A]"
                  }`}
                  style={{ width: `${row.pct}%` }}
                />
              </div>
              <span
                className={`font-mono text-[11px] md:text-[13px] w-14 md:w-16 text-right ${
                  row.top ? "text-foreground font-semibold" : "text-secondary"
                }`}
              >
                {row.pct.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
        <p className="font-mono text-[10px] text-secondary mt-4">
          Model: <Tip term="XGBoost">XGB</Tip>×3 + <Tip term="Random Forest">RF</Tip>×1 · Features: <Tip term="EA FC">EA FC</Tip> + <Tip term="ELO">ELO</Tip> + Form
        </p>
      </section>

      {/* Quick Predict — Head to Head */}
      <section className="flex flex-col lg:flex-row items-center gap-8 lg:gap-16 px-4 md:px-12 lg:px-20 py-10 lg:py-16 bg-[#0D0D0D] border-t border-border">
        {/* Left text */}
        <div className="flex flex-col gap-3 text-center lg:text-left">
          <p className="text-purple text-[11px] font-semibold tracking-[3px]">
            QUICK PREDICTION
          </p>
          <h2 className="font-[family-name:var(--font-anton)] text-[32px] lg:text-[44px] leading-tight">
            HEAD TO HEAD
          </h2>
          <p className="text-secondary text-sm max-w-[280px] mx-auto lg:mx-0">
            Select any two World Cup nations.
            <br />
            Flags animate as probabilities compute.
          </p>
        </div>

        {/* Right team selector */}
        <div className="flex items-center gap-4 md:gap-6">
          {/* Home team box */}
          <div className="flex flex-col items-center gap-2.5 md:gap-3.5 bg-[#141414] border-2 border-[#1A1A1A] rounded-xl px-4 md:px-8 py-5 md:py-7 w-[140px] md:w-[200px]">
            <span className="fi fi-es text-3xl md:text-5xl" />
            <span className="text-base md:text-xl font-bold">Spain</span>
            <span className="text-secondary text-[10px] md:text-[11px]">HOME</span>
          </div>

          {/* VS */}
          <span className="font-[family-name:var(--font-anton)] text-[28px] md:text-[40px] text-purple">
            VS
          </span>

          {/* Away team box */}
          <div className="flex flex-col items-center gap-2.5 md:gap-3.5 bg-[#141414] border-2 border-[#1A1A1A] rounded-xl px-4 md:px-8 py-5 md:py-7 w-[140px] md:w-[200px]">
            <span className="fi fi-ar text-3xl md:text-5xl" />
            <span className="text-base md:text-xl font-bold">Argentina</span>
            <span className="text-secondary text-[10px] md:text-[11px]">AWAY</span>
          </div>

          {/* Predict button */}
          <Link
            href="/predict"
            className="flex items-center justify-center w-12 h-12 md:w-16 md:h-16 rounded-full bg-gradient-to-b from-purple to-pink hover:opacity-80 transition-opacity"
          >
            <span className="text-white text-xl md:text-2xl">→</span>
          </Link>
        </div>
      </section>
    </div>
  );
}
