import Link from "next/link";

const CHART_DATA = [
  { rank: "01", team: "Spain", code: "es", pct: 19.52, top: true },
  { rank: "02", team: "France", code: "fr", pct: 11.68, top: true },
  { rank: "03", team: "Argentina", code: "ar", pct: 8.62, top: true },
  { rank: "04", team: "England", code: "gb-eng", pct: 8.06, top: true },
  { rank: "05", team: "Mexico", code: "mx", pct: 7.58, top: true },
  { rank: "06", team: "Germany", code: "de", pct: 6.93, top: false },
  { rank: "07", team: "Brazil", code: "br", pct: 6.67, top: false },
  { rank: "08", team: "Switzerland", code: "ch", pct: 4.85, top: false },
  { rank: "09", team: "Norway", code: "no", pct: 4.60, top: false },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center h-[280px] px-20 gap-4">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          AI-POWERED TOURNAMENT SIMULATION
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[72px] leading-[1] max-w-[860px]">
          WHO WINS THE
          <br />
          2026 WORLD CUP?
        </h1>
        <p className="text-secondary text-[15px] max-w-[820px]">
          Trained on 60,000+ international matches &middot; 10,000 Monte Carlo
          simulations &middot; 87.5% backtest accuracy
        </p>
        <div className="w-[60px] h-1 bg-purple rounded-sm" />
      </section>

      {/* Championship Probability */}
      <section className="px-20 py-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="font-[family-name:var(--font-anton)] text-[28px] tracking-wide">
            CHAMPIONSHIP PROBABILITY
          </h2>
          <p className="font-mono text-secondary text-xs">
            10,000 tournament simulations · DC×1 + XGB×4 + RF×1
          </p>
        </div>

        <div className="flex flex-col gap-3">
          {CHART_DATA.map((row) => (
            <div key={row.rank} className="flex items-center gap-4 h-10">
              <span
                className={`font-mono text-xs w-6 ${
                  row.top ? "text-purple" : "text-secondary"
                }`}
              >
                {row.rank}
              </span>
              <span className={`fi fi-${row.code} text-lg`} />
              <span
                className={`text-sm font-semibold w-28 ${
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
                className={`font-mono text-[13px] w-16 text-right ${
                  row.top ? "text-foreground font-semibold" : "text-secondary"
                }`}
              >
                {row.pct.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Quick Predict — Head to Head */}
      <section className="flex items-center gap-16 px-20 py-16 bg-[#0D0D0D] border-t border-border">
        {/* Left text */}
        <div className="flex flex-col gap-3">
          <p className="text-purple text-[11px] font-semibold tracking-[3px]">
            QUICK PREDICTION
          </p>
          <h2 className="font-[family-name:var(--font-anton)] text-[44px] leading-tight">
            HEAD TO HEAD
          </h2>
          <p className="text-secondary text-sm max-w-[280px]">
            Select any two World Cup nations.
            <br />
            Flags animate as probabilities compute.
          </p>
        </div>

        {/* Right team selector */}
        <div className="flex items-center gap-6">
          {/* Home team box */}
          <div className="flex flex-col items-center gap-3.5 bg-[#141414] border-2 border-[#1A1A1A] rounded-xl px-8 py-7 w-[200px]">
            <span className="fi fi-es text-5xl" />
            <span className="text-xl font-bold">Spain</span>
            <span className="text-secondary text-[11px]">HOME</span>
          </div>

          {/* VS */}
          <span className="font-[family-name:var(--font-anton)] text-[40px] text-purple">
            VS
          </span>

          {/* Away team box */}
          <div className="flex flex-col items-center gap-3.5 bg-[#141414] border-2 border-[#1A1A1A] rounded-xl px-8 py-7 w-[200px]">
            <span className="fi fi-ar text-5xl" />
            <span className="text-xl font-bold">Argentina</span>
            <span className="text-secondary text-[11px]">AWAY</span>
          </div>

          {/* Predict button */}
          <Link
            href="/predict"
            className="flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-b from-purple to-pink hover:opacity-80 transition-opacity"
          >
            <span className="text-white text-2xl">→</span>
          </Link>
        </div>
      </section>
    </div>
  );
}
