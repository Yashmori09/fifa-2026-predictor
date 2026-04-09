import Image from "next/image";

const SIGNALS = [
  {
    num: "01",
    numColor: "text-purple",
    title: "ELO Ratings",
    desc: "A chess-inspired rating system adapted for international football. Every team starts with a base rating that goes up after wins and down after losses \u2014 weighted by opponent strength and match importance. Teams that consistently beat strong opponents rise to the top.",
    stat: "30.2% of model importance",
    statColor: "text-purple",
  },
  {
    num: "02",
    numColor: "text-cyan",
    title: "EA FC Squad Ratings",
    desc: "Player ratings from EA Sports FC (formerly FIFA) video games. Professional scouts rate every player\u2019s pace, shooting, passing, defending, and physicality on a 1\u201399 scale. We aggregate these into squad-level metrics \u2014 overall strength, positional depth, and top-player quality.",
    stat: "31.2% of model importance",
    statColor: "text-cyan",
  },
  {
    num: "03",
    numColor: "text-pink",
    title: "Form, History & Context",
    desc: "Recent win rate, goals scored and conceded, head-to-head record between the two teams, tournament importance (World Cup vs friendly), home advantage, and momentum \u2014 whether a team is on a winning or losing streak heading into the match.",
    stat: "38.6% of model importance",
    statColor: "text-pink",
  },
];

const MODEL_CARDS = [
  {
    tag: "\u00d7 3 COPIES",
    tagColor: "text-purple",
    borderColor: "border-purple",
    title: "XGBoost",
    desc: "Gradient-boosted decision trees. Learns complex non-linear patterns \u2014 like how ELO difference matters more in knockout rounds than group stages. Handles missing data natively.",
  },
  {
    tag: "\u00d7 1 COPY",
    tagColor: "text-cyan",
    borderColor: "border-cyan",
    title: "Random Forest",
    desc: "500 independent decision trees that each see a random subset of features. Provides stability and reduces overfitting \u2014 when XGBoost is too confident, the Random Forest pulls predictions back toward reality.",
  },
];

const STATS = [
  { value: "97", label: "Input Features", color: "text-purple" },
  { value: "35,304", label: "Training Matches", color: "text-cyan" },
  { value: "1884\u20132024", label: "Training Period", color: "text-pink" },
  { value: "10,000", label: "Tournament Sims", color: "text-green-500" },
];

const STEPS = [
  {
    num: "STEP 1",
    numColor: "text-purple",
    title: "Predict Outcome Probabilities",
    desc: "The ensemble outputs three probabilities for each match. Example: France vs Brazil \u2192 45% home win, 28% draw, 27% away win.",
  },
  {
    num: "STEP 2",
    numColor: "text-cyan",
    title: "Reverse-Engineer Goal Rates",
    desc: "Using a precomputed Poisson grid, we find the goal-scoring rates (\u03bb) that best reproduce those probabilities. If France has a 45% win chance, we find \u03bb values where Poisson-generated scores give ~45% home wins.",
  },
  {
    num: "STEP 3",
    numColor: "text-pink",
    title: "Simulate Scorelines",
    desc: "Random goals are drawn from the Poisson distribution \u2014 so the same match might be 2-1 in one simulation and 0-0 in another. Over 10,000 runs, the randomness averages out and the true probabilities emerge.",
  },
];

const DISCOVERIES = [
  [
    {
      tag: "THE ECHO CHAMBER PROBLEM",
      tagColor: "text-red-500",
      title: "Win-based ratings inflate weak-region teams",
      desc: "ELO ratings are calculated from match results. Teams in weaker regions (like CONCACAF or AFC) accumulate inflated ratings by beating other weak teams. Mexico and Japan appeared as realistic contenders in early models \u2014 anyone who watches football knows that\u2019s wrong. Adding scout-assessed player ratings from EA FC broke the echo chamber.",
    },
    {
      tag: "THE COVERAGE TRADE-OFF",
      tagColor: "text-amber-500",
      title: "More matches > better features",
      desc: "EA player ratings only exist from 2014, but our training data goes back to 1884. We tried training only on modern data (6,000 matches with full features) vs all data (35,000 matches with gaps). The larger dataset won every time \u2014 old matches still teach the model about ELO patterns, form, and home advantage.",
    },
  ],
  [
    {
      tag: "CALIBRATION > RAW ACCURACY",
      tagColor: "text-cyan",
      title: "A well-calibrated model matters more for simulation",
      desc: "When the model says \u201870% chance of winning,\u2019 does it actually happen 70% of the time? That\u2019s calibration. For Monte Carlo simulation, calibration matters more than getting individual matches right \u2014 you want the dice to be fair, even if you can\u2019t predict every roll. Our model achieves ECE of 0.018 (near-perfect calibration).",
    },
    {
      tag: "LEAN BEATS COMPLEX",
      tagColor: "text-green-500",
      title: "9 difference features outperform 44 detailed features",
      desc: "We tried feeding the model 44 individual squad attributes (pace, shooting, defending for each team). But simple difference features \u2014 \u2018how much better is Team A\u2019s attack than Team B\u2019s defense?\u2019 \u2014 performed equally well with far less noise. When 82% of training rows have missing squad data, simpler is better.",
    },
  ],
  [
    {
      tag: "FOOTBALL HAS EVOLVED",
      tagColor: "text-purple",
      title: "A 3-0 win in 1920 is not the same as a 3-0 win today",
      desc: "Goals per game have dropped from 5.5 to 2.7 over 140 years. Home advantage has shrunk from 40% to 23%. The game has fundamentally changed \u2014 tactics are more defensive, away teams are better prepared, and international squads are more balanced. The model is trained on all eras but learns to weight recent patterns more heavily.",
    },
    {
      tag: "THE 17% THAT MATTERS",
      tagColor: "text-pink",
      title: "ELO and EA ratings agree 83% of the time \u2014 the value is where they disagree",
      desc: "Both ELO and EA FC ratings correctly rank France above Norway and Brazil above Bolivia. That\u2019s the easy 83%. The real value of adding squad ratings is the 17% where they disagree \u2014 teams like Mexico and Japan whose ELO is inflated by beating weak opponents, but whose player ratings reveal they lack the individual quality to compete with Europe and South America\u2019s best.",
    },
  ],
];

const BIAS_ROWS = [
  { team: "Mexico", before: "6.28%", after: "0.31%", delta: "\u22125.97" },
  { team: "Senegal", before: "6.76%", after: "1.55%", delta: "\u22125.21" },
  { team: "Japan", before: "6.27%", after: "0.18%", delta: "\u22126.09" },
  { team: "Australia", before: "5.19%", after: "0.06%", delta: "\u22125.13" },
];

export default function MethodologyPage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="flex flex-col justify-center px-4 md:px-12 lg:px-20 py-8 md:py-12 gap-3.5">
        <p className="text-purple text-[11px] font-semibold tracking-[3px]">
          METHODOLOGY
        </p>
        <h1 className="font-[family-name:var(--font-anton)] text-[36px] md:text-[48px] lg:text-[64px] leading-[1.05]">
          HOW WE PREDICT
          <br />
          THE WORLD CUP
        </h1>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[700px]">
          A machine learning ensemble trained on 35,000+ international matches,
          combining team strength ratings, squad quality data, and historical
          form to simulate the entire tournament 10,000 times.
        </p>
      </section>

      {/* The Three Signals */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          THE THREE SIGNALS
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[600px] mb-6 md:mb-8">
          Our model combines three independent sources of team strength. Each
          captures something the others miss.
        </p>
        <div className="flex flex-col md:flex-row gap-4">
          {SIGNALS.map((s) => (
            <div
              key={s.num}
              className="flex-1 flex flex-col gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-7 min-h-0 md:min-h-[260px]"
            >
              <span className={`font-mono text-xs font-semibold ${s.numColor}`}>
                {s.num}
              </span>
              <span className="text-lg font-bold">{s.title}</span>
              <p className="text-[13px] text-[#A1A1AA] leading-[1.7] flex-1">
                {s.desc}
              </p>
              <span
                className={`font-mono text-[13px] font-semibold ${s.statColor}`}
              >
                {s.stat}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* The Model */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          THE MODEL
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-8">
          Instead of one model making all decisions, we use an ensemble — four
          models that each learn different patterns, then vote on the outcome.
        </p>

        {/* Model cards */}
        <div className="flex flex-col md:flex-row gap-3 mb-6 md:mb-8">
          {MODEL_CARDS.map((m) => (
            <div
              key={m.title}
              className={`flex-1 flex flex-col gap-2.5 bg-[#111111] border ${m.borderColor} rounded-xl p-5 md:p-6 min-h-0 md:min-h-[200px]`}
            >
              <span
                className={`font-mono text-xs font-semibold tracking-wider ${m.tagColor}`}
              >
                {m.tag}
              </span>
              <span className="text-xl font-bold">{m.title}</span>
              <p className="text-[13px] text-[#A1A1AA] leading-[1.7]">
                {m.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Stat row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="flex flex-col gap-1 bg-[#0D0D0D] border border-[#1A1A1A] rounded-lg p-4 md:p-5"
            >
              <span
                className={`font-[family-name:var(--font-anton)] text-2xl md:text-4xl ${s.color}`}
              >
                {s.value}
              </span>
              <span className="text-[12px] md:text-[13px] text-secondary">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* From Win Probability to Scorelines */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          FROM WIN PROBABILITY TO SCORELINES
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-7">
          The model predicts the probability of home win, draw, or away win. But
          to simulate a tournament, we need actual scorelines. Here&apos;s how we
          bridge that gap.
        </p>
        <div className="flex flex-col md:flex-row gap-4">
          {STEPS.map((s) => (
            <div
              key={s.num}
              className="flex-1 flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6 min-h-0 md:min-h-[190px]"
            >
              <span
                className={`font-mono text-xs font-semibold tracking-wider ${s.numColor}`}
              >
                {s.num}
              </span>
              <span className="text-base font-bold">{s.title}</span>
              <p className="text-[13px] text-[#A1A1AA] leading-[1.7]">
                {s.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* What We Discovered */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          WHAT WE DISCOVERED
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[600px] mb-6 md:mb-8">
          Building this model taught us things about football prediction that
          weren&apos;t obvious at the start.
        </p>
        <div className="flex flex-col gap-4">
          {DISCOVERIES.map((row, ri) => (
            <div key={ri} className="flex flex-col md:flex-row gap-4">
              {row.map((d) => (
                <div
                  key={d.tag}
                  className="flex-1 flex flex-col gap-3 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-7"
                >
                  <span
                    className={`font-mono text-[10px] md:text-xs font-semibold tracking-[2px] ${d.tagColor}`}
                  >
                    {d.tag}
                  </span>
                  <span className="text-base font-bold">{d.title}</span>
                  <p className="text-[13px] text-[#A1A1AA] leading-[1.7]">
                    {d.desc}
                  </p>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* Model Accuracy */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          MODEL ACCURACY
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-8">
          Tested on 3,313 matches from 2023&ndash;2024 that the model never saw
          during training, and backtested against the 2022 World Cup.
        </p>

        {/* Stat cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6 md:mb-8">
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-purple">
              0.018
            </span>
            <span className="text-[13px] text-secondary">
              Expected Calibration Error
            </span>
            <span className="text-xs text-[#A1A1AA]">
              Near-perfect probability accuracy
            </span>
          </div>
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-cyan">
              1.3%
            </span>
            <span className="text-[13px] text-secondary">
              Maximum Prediction Bias
            </span>
            <span className="text-xs text-[#A1A1AA]">
              Home win slightly overestimated
            </span>
          </div>
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-pink">
              0.826
            </span>
            <span className="text-[13px] text-secondary">Log Loss</span>
            <span className="text-xs text-[#A1A1AA]">
              Lower is better &mdash; measures prediction confidence
            </span>
          </div>
        </div>

        {/* Calibration images */}
        <div className="flex flex-col gap-5">
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/phase2_05_calibration_curves.png"
              alt="Calibration curves showing predicted vs actual probabilities"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/phase2_05_reliability_diagrams.png"
              alt="Reliability diagrams with sample counts"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
        </div>
      </section>

      {/* 2022 World Cup Backtest */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          2022 WORLD CUP BACKTEST
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-7">
          We ran 10,000 simulations of the 2022 Qatar World Cup to see how the
          model would have performed. It correctly identified Argentina as a top
          contender with 24.5% championship probability.
        </p>

        <div className="flex flex-col md:flex-row gap-4">
          {/* Left — champion & runner-up */}
          <div className="flex-1 flex flex-col gap-3">
            <div className="flex flex-col gap-2 bg-[#111111] border border-purple rounded-xl p-5 md:p-6">
              <span className="font-mono text-xs font-semibold tracking-[2px] text-purple">
                CHAMPION &mdash; ARGENTINA
              </span>
              <span className="text-base font-bold">
                Ranked #2 &mdash; 24.51% win probability
              </span>
              <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
                Behind only Spain (25.6%) &mdash; correctly identified as a
                top-2 contender
              </p>
            </div>
            <div className="flex flex-col gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
              <span className="font-mono text-xs font-semibold tracking-[2px] text-cyan">
                RUNNER-UP &mdash; FRANCE
              </span>
              <span className="text-base font-bold">
                Ranked #4 &mdash; 11.89% win probability
              </span>
              <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
                In the top tier &mdash; correctly flagged as a serious contender
              </p>
            </div>
          </div>

          {/* Right — bias correction */}
          <div className="flex-1">
            <div className="flex flex-col gap-3.5 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-7 h-full">
              <span className="font-mono text-xs font-semibold tracking-[2px] text-green-500">
                BIAS CORRECTION VALIDATED
              </span>
              <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
                Teams that were unrealistically ranked in early models got
                corrected:
              </p>
              {BIAS_ROWS.map((r) => (
                <div key={r.team} className="flex items-center gap-2">
                  <span className="text-[13px] w-[80px] md:w-[120px]">{r.team}</span>
                  <span className="font-mono text-[11px] md:text-xs text-green-500 flex-1">
                    {r.before} &rarr; {r.after}
                  </span>
                  <span className="font-mono text-[11px] md:text-xs font-bold text-green-500">
                    {r.delta}
                  </span>
                </div>
              ))}
              <p className="text-xs text-secondary leading-relaxed mt-1">
                Adding EA squad ratings &mdash; assessed by scouts, not match
                results &mdash; corrected the inflated rankings of teams from
                weaker confederations.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
