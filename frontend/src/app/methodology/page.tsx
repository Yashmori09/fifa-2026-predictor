import Image from "next/image";

const ARCHITECTURE_BLOCKS = [
  {
    title: "XGBoost",
    weight: "×4 weight",
    desc: "59 features\nELO + Form + H2H + DC",
    borderColor: "border-purple",
    titleColor: "text-purple",
  },
  {
    title: "Random Forest",
    weight: "×1 weight",
    desc: "59 features\nIsotonic calibrated",
    borderColor: "border-pink",
    titleColor: "text-pink",
  },
  {
    title: "Dixon-Coles",
    weight: "Train ×5 | Sim ×1",
    desc: "Bivariate Poisson\n38,617 matches",
    borderColor: "border-cyan",
    titleColor: "text-cyan",
  },
  {
    title: "Blended\nEnsemble",
    weight: "Log Loss: 0.7988",
    desc: "Soft probability\nblend",
    borderColor: "border-purple",
    titleColor: "text-foreground",
    gradient: true,
  },
];

const EXPERIMENTS = [
  { name: "ELO + Form", ll: "0.8333", barH: 100, color: "bg-[#2A2A2A]" },
  { name: "+ H2H & Venue", ll: "0.8263", barH: 80, color: "bg-[#3A2A4A]" },
  { name: "+ Dixon-Coles", ll: "0.8131", barH: 55, color: "bg-[#5A2A7A]" },
  { name: "+ Hypertuning", ll: "0.8123", barH: 50, color: "bg-[#7A2A9A]" },
  { name: "+ DC as Voter", ll: "0.7988", barH: 20, color: "bg-purple", highlight: true },
];

const KEY_FINDINGS = [
  {
    title: "ELO is King",
    titleColor: "text-purple",
    iconColor: "bg-purple",
    desc: "ELO diff has 0.476 correlation with match outcome — the single strongest predictor in the model",
  },
  {
    title: "DC as Direct Voter",
    titleColor: "text-cyan",
    iconColor: "bg-cyan",
    desc: "Adding Dixon-Coles as a direct probability voter (not just features) gave the biggest single improvement: −0.0135",
  },
  {
    title: "Well Calibrated",
    titleColor: "text-pink",
    iconColor: "bg-pink",
    desc: "ECE of 0.027 — when the model says 70% win, it happens ~70% of the time. No post-hoc recalibration needed",
  },
  {
    title: "Cross-Confederation",
    titleColor: "text-purple",
    iconColor: "bg-purple",
    desc: "DC can't compare teams across confederations — ELO solves this. Use each model for what it's actually good at",
  },
];

const FEATURES = [
  { name: "elo_diff_sq", importance: 0.1855 },
  { name: "elo_diff", importance: 0.1400 },
  { name: "h2h_recent_win_rate", importance: 0.0220 },
  { name: "dc_goal_diff", importance: 0.0194 },
  { name: "away_conf_CAF", importance: 0.0187 },
  { name: "net_goal_diff", importance: 0.0186 },
  { name: "dc_away_win_prob", importance: 0.0167 },
  { name: "h2h_home_win_rate", importance: 0.0154 },
];

const FEATURE_EXPLANATIONS = [
  {
    title: "ELO Difference",
    color: "border-purple",
    titleColor: "text-purple",
    desc: "Overall team strength gap. Higher = more dominant team. Best cross-confederation signal.",
  },
  {
    title: "Dixon-Coles Probs + xG",
    color: "border-cyan",
    titleColor: "text-cyan",
    desc: "Bivariate Poisson model predicting exact scorelines. Captures goal-scoring dominance ELO misses.",
  },
  {
    title: "Recent Form (5 & 10 games)",
    color: "border-pink",
    titleColor: "text-pink",
    desc: "Win rate, goals scored/conceded, goal diff trend. Captures momentum — are they hot right now?",
  },
  {
    title: "H2H Win Rate",
    color: "border-border",
    titleColor: "text-foreground",
    desc: "Historical head-to-head record between these two nations. Confidence-weighted by number of matches.",
  },
];

export default function MethodologyPage() {
  const maxImp = Math.max(...FEATURES.map((f) => f.importance));

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center h-[160px] px-20 gap-2.5">
        <p className="text-cyan text-[11px] font-semibold tracking-[3px]">
          RESEARCH & EXPERIMENTS
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[48px] leading-tight">
          HOW THE MODEL WORKS
        </h1>
        <p className="text-secondary text-sm">
          30 experiments across 5 phases — from baseline ensemble to
          Dixon-Coles integration
        </p>
      </section>

      {/* Architecture */}
      <section className="px-20 py-12">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
          MODEL ARCHITECTURE
        </h2>
        <div className="flex items-center gap-2">
          {ARCHITECTURE_BLOCKS.map((block, i) => (
            <div key={block.title} className="flex items-center gap-2 flex-1">
              <div
                className={`flex-1 flex flex-col items-center gap-2.5 rounded-lg px-6 py-5 border ${
                  block.gradient
                    ? "bg-[#0D0D18] border-2"
                    : `bg-[#141414] ${block.borderColor}`
                }`}
                style={
                  block.gradient
                    ? {
                        borderImage:
                          "linear-gradient(180deg, #A855F7, #EC4899, #06B6D4) 1",
                      }
                    : undefined
                }
              >
                <span
                  className={`font-[family-name:var(--font-anton)] text-base text-center whitespace-pre-line ${block.titleColor}`}
                >
                  {block.title}
                </span>
                <span className="font-mono text-[11px] text-secondary text-center">
                  {block.weight}
                </span>
                <span className="text-[11px] text-secondary text-center whitespace-pre-line">
                  {block.desc}
                </span>
              </div>
              {i < ARCHITECTURE_BLOCKS.length - 1 && (
                <span className="text-purple text-2xl">→</span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Experiment Progression */}
      <section className="px-20 py-10 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-1">
          EXPERIMENT PROGRESSION
        </h2>
        <p className="text-secondary text-[13px] mb-6">
          30 experiments across 5 phases — Log Loss improved 0.8333 → 0.7988
          (−0.0345)
        </p>
        <div className="flex items-end gap-2 h-[150px]">
          {EXPERIMENTS.map((exp) => (
            <div
              key={exp.name}
              className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end"
            >
              <div
                className={`w-full rounded-t-[4px] ${exp.color}`}
                style={{ height: `${exp.barH}px` }}
              />
              <span
                className={`font-mono text-[9px] whitespace-pre-line text-center leading-tight ${
                  exp.highlight ? "text-purple" : "text-secondary"
                }`}
              >
                {exp.name}
                {"\n"}
                {exp.ll}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Key Findings */}
      <section className="px-20 py-10">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-6">
          KEY FINDINGS
        </h2>
        <div className="flex gap-4">
          {KEY_FINDINGS.map((finding) => (
            <div
              key={finding.title}
              className="flex-1 bg-[#111111] border border-[#1A1A1A] rounded-lg p-6 flex flex-col gap-3"
            >
              <div className={`w-8 h-1 rounded-full ${finding.iconColor}`} />
              <h3
                className={`font-[family-name:var(--font-anton)] text-base ${finding.titleColor}`}
              >
                {finding.title}
              </h3>
              <p className="text-xs text-secondary leading-relaxed">
                {finding.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Feature Importance */}
      <section className="px-20 py-12 border-t border-border">
        <div className="flex gap-12">
          {/* Left — bars */}
          <div className="w-[580px] flex flex-col gap-5">
            <div>
              <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-1">
                TOP FEATURES
              </h2>
              <p className="text-secondary text-[13px]">
                What drives the prediction — XGBoost feature importance
              </p>
            </div>
            <div className="flex flex-col gap-2.5">
              {FEATURES.map((feat) => (
                <div key={feat.name} className="flex items-center gap-3 h-8">
                  <span className="font-mono text-[11px] w-[170px] shrink-0 text-secondary">{feat.name}</span>
                  <div className="flex-1 h-5 bg-[#1A1A1A] rounded overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple to-pink rounded"
                      style={{ width: `${(feat.importance / maxImp) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs text-purple w-10 text-right">
                    {(feat.importance * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Right — explanations */}
          <div className="flex-1 flex flex-col gap-4">
            <h2 className="font-[family-name:var(--font-anton)] text-lg tracking-wide">
              WHAT EACH FEATURE TELLS US
            </h2>
            {FEATURE_EXPLANATIONS.map((exp) => (
              <div
                key={exp.title}
                className={`bg-[#111111] border ${exp.color} rounded-lg px-[18px] py-4 flex flex-col gap-1.5`}
              >
                <span className={`text-xs font-bold ${exp.titleColor}`}>
                  {exp.title}
                </span>
                <span className="text-[11px] text-secondary leading-relaxed">
                  {exp.desc}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Calibration */}
      <section className="px-20 py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-2xl tracking-wide mb-1">
          CALIBRATION RESULTS
        </h2>
        <p className="text-secondary text-[13px] mb-5">
          When the model says 70% win — does it actually happen 70% of the time?
        </p>

        {/* Stat cards */}
        <div className="flex gap-3 mb-8">
          <div className="flex-1 bg-[#111111] border border-purple rounded-lg p-5 flex flex-col items-center gap-2">
            <p className="text-[10px] text-secondary font-semibold tracking-[2px]">
              MEAN ECE
            </p>
            <p className="font-[family-name:var(--font-anton)] text-[32px] text-purple">
              0.027
            </p>
            <p className="text-[11px] text-secondary text-center">
              Well-calibrated &lt; 0.05 threshold
            </p>
          </div>
          <div className="flex-1 bg-[#111111] border border-cyan rounded-lg p-5 flex flex-col items-center gap-2">
            <p className="text-[10px] text-secondary font-semibold tracking-[2px]">
              MAX BIAS
            </p>
            <p className="font-[family-name:var(--font-anton)] text-[32px] text-cyan">
              1.2%
            </p>
            <p className="text-[11px] text-secondary text-center">
              Draws see slight underprediction
            </p>
          </div>
          <div className="flex-1 bg-[#111111] border border-pink rounded-lg p-5 flex flex-col items-center gap-2">
            <p className="text-[10px] text-secondary font-semibold tracking-[2px]">
              LOG LOSS
            </p>
            <p className="font-[family-name:var(--font-anton)] text-[32px] text-pink">
              0.7988
            </p>
            <p className="text-[11px] text-secondary text-center">
              −0.0345 from start
            </p>
          </div>
        </div>

        {/* Calibration Curves */}
        <div className="mb-5">
          <h3 className="font-[family-name:var(--font-anton)] text-lg mb-0.5 tracking-wide">
            CALIBRATION CURVES
          </h3>
          <p className="text-[11px] text-secondary mb-3">
            Predicted probability vs actual outcome rate — 10 uniform bins per
            class
          </p>
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/calibration_curves.png"
              alt="Calibration curves showing predicted vs actual probabilities"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
        </div>

        {/* Reliability Diagrams */}
        <div className="mb-5">
          <h3 className="font-[family-name:var(--font-anton)] text-lg mb-0.5 tracking-wide">
            RELIABILITY DIAGRAMS
          </h3>
          <p className="text-[11px] text-secondary mb-3">
            Dot size = sample count per bin — larger dot means more matches back
            that probability range
          </p>
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/reliability_diagrams.png"
              alt="Reliability diagrams with sample counts"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
        </div>

        {/* Verdict */}
        <div className="flex items-center gap-4 bg-[#111111] border border-purple rounded-lg px-5 py-4">
          <span className="text-purple text-xl font-bold">✓</span>
          <p className="text-[13px]">
            No post-hoc recalibration needed — the blended ensemble is
            inherently well-calibrated. Simulation probabilities are
            trustworthy.
          </p>
        </div>
      </section>
    </div>
  );
}
