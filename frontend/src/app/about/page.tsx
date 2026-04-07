const PROJECT_STATS = [
  { label: "Log Loss", value: "0.7988", color: "text-purple" },
  { label: "Backtest Accuracy", value: "87.5%", color: "text-cyan" },
  { label: "Calibration (ECE)", value: "0.027", color: "text-pink" },
  { label: "Simulations Run", value: "10,000", color: "text-purple" },
  { label: "Total Matches", value: "60,000+", color: "text-cyan" },
];

const PHASE2_CARDS = [
  {
    icon: "◈",
    iconColor: "text-purple",
    title: "Squad Intelligence",
    desc: "Current form of every\nplayer in the 2026 squad",
    tagColor: "text-purple",
    tagBg: "bg-[#1A0D2A]",
    borderColor: "border-[#2A1A3A]",
  },
  {
    icon: "◈",
    iconColor: "text-cyan",
    title: "Star Power Index",
    desc: "Some players change\ngames. The model will know.",
    tagColor: "text-cyan",
    tagBg: "bg-[#0A1E1E]",
    borderColor: "border-[#0D2A2A]",
  },
  {
    icon: "◈",
    iconColor: "text-pink",
    title: "Live Squad Updates",
    desc: "2026 rosters. Injuries.\nForm going into the tournament.",
    tagColor: "text-pink",
    tagBg: "bg-[#1E0A12]",
    borderColor: "border-[#2A0D1A]",
  },
];

export default function AboutPage() {
  return (
    <div className="flex flex-col">
      {/* Hero — two column */}
      <section className="flex items-center gap-20 px-20 py-20">
        {/* Left */}
        <div className="flex flex-col gap-5 w-[560px]">
          <p className="text-purple text-[11px] font-semibold tracking-[3px]">
            BUILT BY
          </p>
          <h1 className="font-[family-name:var(--font-anton)] text-[64px] leading-none">
            YASH MORI
          </h1>
          <p className="text-secondary text-base">
            AI Engineer · Bangalore, India
          </p>
          <p className="text-secondary text-sm leading-relaxed max-w-[520px]">
            Built this project to explore the intersection of sports analytics
            and machine learning — from raw match data to a fully simulated 2026
            World Cup using Dixon-Coles + ELO ensemble models.
          </p>
          <div className="flex gap-3">
            <a
              href="https://github.com/Yashmori09"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-5 py-2.5 bg-[#141414] border border-[#1A1A1A] rounded text-[13px] font-medium hover:border-secondary transition-colors"
            >
              GitHub →
            </a>
            <a
              href="https://linkedin.com/in/YashMori"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-5 py-2.5 bg-purple rounded text-[13px] font-medium hover:opacity-90 transition-opacity"
            >
              LinkedIn →
            </a>
          </div>
        </div>

        {/* Right — Project Stats */}
        <div className="flex-1 flex flex-col gap-4">
          <h2 className="font-[family-name:var(--font-anton)] text-base tracking-wide">
            PROJECT STATS
          </h2>
          {PROJECT_STATS.map((stat) => (
            <div
              key={stat.label}
              className="flex items-center justify-between bg-[#111111] border border-[#1A1A1A] rounded-lg px-5 py-4"
            >
              <span className="text-secondary text-[13px]">{stat.label}</span>
              <span className={`font-mono text-base font-semibold ${stat.color}`}>
                {stat.value}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Phase 2 Teaser */}
      <section className="flex flex-col items-center gap-8 px-20 py-[60px] bg-[#050505] border-t border-border">
        <p className="text-purple text-[11px] font-semibold tracking-[4px]">
          PHASE 2 — IN DEVELOPMENT
        </p>
        <h2 className="font-[family-name:var(--font-anton)] text-[52px] leading-tight">
          THE MODEL LEARNS PLAYERS
        </h2>
        <p className="text-secondary text-[15px] text-center max-w-[600px]">
          Individual brilliance. Squad depth. The 26 players who actually take
          the field in 2026.
        </p>
        <div className="flex gap-4 w-full">
          {PHASE2_CARDS.map((card) => (
            <div
              key={card.title}
              className={`flex-1 flex flex-col items-center gap-3 bg-[#0F0F0F] border ${card.borderColor} rounded-lg px-5 py-6`}
            >
              <span className={`text-[28px] ${card.iconColor}`}>
                {card.icon}
              </span>
              <span className="font-[family-name:var(--font-anton)] text-lg">
                {card.title}
              </span>
              <span className="text-secondary text-xs text-center whitespace-pre-line leading-relaxed">
                {card.desc}
              </span>
              <span
                className={`font-mono text-[10px] tracking-[2px] ${card.tagColor} ${card.tagBg} rounded-full px-3 py-1`}
              >
                LOCKED
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="flex items-center justify-center h-[60px] bg-[#050505] border-t border-border">
        <span className="text-secondary text-xs">
          FIFA 2026 Predictor · Built by Yash Mori · AI Engineer
        </span>
      </footer>
    </div>
  );
}
