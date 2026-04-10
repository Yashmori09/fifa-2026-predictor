const PROJECT_STATS = [
  { label: "Calibration (ECE)", value: "0.018", color: "text-purple" },
  { label: "Log Loss", value: "0.826", color: "text-cyan" },
  { label: "Input Features", value: "97", color: "text-pink" },
  { label: "Training Matches", value: "35,304", color: "text-purple" },
  { label: "Training Span", value: "1884–2024", color: "text-cyan" },
  { label: "Simulations", value: "10,000", color: "text-pink" },
];

export default function AboutPage() {
  return (
    <div className="flex flex-col">
      {/* Hero — two column */}
      <section className="flex flex-col lg:flex-row items-start lg:items-center gap-10 lg:gap-20 px-4 md:px-12 lg:px-20 py-10 md:py-16 lg:py-20">
        {/* Left */}
        <div className="flex flex-col gap-5 w-full lg:w-[560px]">
          <p className="text-purple text-[11px] font-semibold tracking-[3px]">
            BUILT BY
          </p>
          <h1 className="font-[family-name:var(--font-anton)] text-[40px] md:text-[52px] lg:text-[64px] leading-none">
            YASH MORI
          </h1>
          <p className="text-secondary text-base">
            AI/ML Engineer &middot; Bangalore, India
          </p>
          <p className="text-secondary text-sm leading-relaxed max-w-[520px]">
            Built this project to explore the intersection of sports analytics
            and machine learning — from raw match data to a fully simulated 2026
            World Cup using an XGBoost + Random Forest ensemble with ELO and EA
            FC squad ratings.
          </p>
          <div className="flex gap-3">
            <a
              href="https://github.com/Yashmori09/fifa-2026-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-5 py-2.5 bg-[#141414] border border-[#1A1A1A] rounded text-[13px] font-medium hover:border-secondary transition-colors"
            >
              GitHub &rarr;
            </a>
            <a
              href="https://www.linkedin.com/in/yash-mori090102/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-5 py-2.5 bg-purple rounded text-[13px] font-medium hover:opacity-90 transition-opacity"
            >
              LinkedIn &rarr;
            </a>
          </div>
        </div>

        {/* Right — Project Stats */}
        <div className="flex-1 w-full flex flex-col gap-4">
          <h2 className="font-[family-name:var(--font-anton)] text-base tracking-wide">
            PROJECT STATS
          </h2>
          {PROJECT_STATS.map((stat) => (
            <div
              key={stat.label}
              className="flex items-center justify-between bg-[#111111] border border-[#1A1A1A] rounded-lg px-4 md:px-5 py-3.5 md:py-4"
            >
              <span className="text-secondary text-[13px]">{stat.label}</span>
              <span
                className={`font-mono text-base font-semibold ${stat.color}`}
              >
                {stat.value}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Tech Stack */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          TECH STACK
        </h2>
        <p className="text-secondary text-xs md:text-sm mb-6 md:mb-8">
          What powers the predictions, the simulations, and the frontend.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex flex-col gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-purple">
              ML &amp; DATA
            </span>
            <div className="flex flex-col gap-2">
              {["XGBoost", "scikit-learn", "NumPy / Pandas", "Jupyter Notebooks"].map((t) => (
                <span key={t} className="text-[13px] text-[#A1A1AA]">{t}</span>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-cyan">
              BACKEND
            </span>
            <div className="flex flex-col gap-2">
              {["Python", "FastAPI", "Poisson Simulation Engine", "HuggingFace Spaces"].map((t) => (
                <span key={t} className="text-[13px] text-[#A1A1AA]">{t}</span>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-pink">
              FRONTEND
            </span>
            <div className="flex flex-col gap-2">
              {["Next.js", "TypeScript", "Tailwind CSS", "Vercel"].map((t) => (
                <span key={t} className="text-[13px] text-[#A1A1AA]">{t}</span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Data Sources */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          DATA SOURCES
        </h2>
        <p className="text-secondary text-xs md:text-sm mb-6 md:mb-8">
          Four independent datasets, combined to give the model a complete picture.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="text-base font-bold">International Match Database</span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              35,304 matches from 1884 to 2024 — every FIFA-recognized international result including friendlies, qualifiers, and tournament matches.
            </p>
          </div>
          <div className="flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="text-base font-bold">ELO Rating System</span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              Historical ELO ratings for every team at the time of each match. Updated after every result, weighted by opponent strength and match importance.
            </p>
          </div>
          <div className="flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="text-base font-bold">EA Sports FC Ratings</span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              Squad-level player ratings from EA FC (FIFA 15 through FC 26). Scout-assessed attributes covering pace, shooting, passing, defending, and physicality.
            </p>
          </div>
          <div className="flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="text-base font-bold">Football Manager 2023</span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              Player attributes for 209 nationalities — calibrated to EA scale to fill coverage gaps for nations not in the EA FC database.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="flex items-center justify-center py-4 md:h-[60px] bg-[#050505] border-t border-border">
        <span className="text-secondary text-xs text-center px-4">
          FIFA 2026 Predictor &middot; Built by Yash Mori &middot; AI/ML Engineer
        </span>
      </footer>
    </div>
  );
}
