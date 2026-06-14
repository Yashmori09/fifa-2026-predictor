import Image from "next/image";
import Tip from "@/components/Tip";

const SIGNALS = [
  {
    num: "01",
    numColor: "text-purple",
    title: "ELO, Form & Match Context",
    desc: "A chess-inspired rating system adapted for international football, plus recent form (last 5 and 10 matches), head-to-head history, confederation, and tournament importance. The model also receives a Dixon-Coles attack/defense rating per team — computed without leaking into the holdout window.",
    stat: "42.3% of model importance",
    statColor: "text-purple",
  },
  {
    num: "02",
    numColor: "text-cyan",
    title: "EA FC Squad Ratings",
    desc: "Per-year squad ratings from EA Sports FC (FIFA 15 through FC 26). Scout-assessed attributes are aggregated per team into overall strength, positional splits (GK / DEF / MID / FWD), top-player quality, depth, and the six core team attributes (pace, shooting, passing, dribbling, defending, physic).",
    stat: "31.3% of model importance",
    statColor: "text-cyan",
  },
  {
    num: "03",
    numColor: "text-pink",
    title: "StatsBomb Intl & Chemistry",
    desc: "New in Phase 3. Event-level data from 314 international tournament matches (WC 2018/2022, Euro 2020/2024, Copa America 2024, AFCON 2023) gives each team an xG profile in international football specifically. Combined with squad chemistry (same-club concentration, average international caps, average age) and 24-month international form.",
    stat: "26.3% of model importance",
    statColor: "text-pink",
  },
];

const MODEL_CARDS = [
  {
    tag: "× 2 REGRESSORS",
    tagColor: "text-purple",
    borderColor: "border-purple",
    title: "XGBoost Poisson",
    desc: "Two gradient-boosted regressors trained with Poisson loss — one predicts the home team's expected goals (λₕ), the other the away team's (λₐ). Predicting goals directly (instead of W/D/L) means draws emerge naturally when the two λ's are close.",
  },
  {
    tag: "SCORELINE ENGINE",
    tagColor: "text-cyan",
    borderColor: "border-cyan",
    title: "Dixon-Coles",
    desc: "The 11×11 joint scoreline matrix P(home=i, away=j) is built from the two Poisson distributions, then corrected for the four low-score cells (0–0, 1–0, 0–1, 1–1) where real-world football diverges from pure Poisson. Match outcome probabilities sum from the matrix.",
  },
];

const STATS = [
  { value: "157", label: "Input Features", color: "text-purple" },
  { value: "6,162", label: "Training Matches", color: "text-cyan" },
  { value: "2018–2025", label: "Training Period", color: "text-pink" },
  { value: "100,000", label: "Tournament Sims", color: "text-green-500" },
];

const STEPS = [
  {
    num: "STEP 1",
    numColor: "text-purple",
    title: "Predict Expected Goals",
    desc: "The two Poisson regressors output λₕ and λₐ directly from the 157 features. Example: Spain vs France → λₕ = 2.16, λₐ = 1.40.",
  },
  {
    num: "STEP 2",
    numColor: "text-cyan",
    title: "Build Scoreline Matrix",
    desc: "Joint probability P(home=i, away=j) for every (i, j) up to 10–10 from the two Poisson distributions. Apply the Dixon-Coles correction to the four low-score cells where real football deviates from independent Poisson.",
  },
  {
    num: "STEP 3",
    numColor: "text-pink",
    title: "Sum to W / D / L",
    desc: "Above the diagonal = home win, on the diagonal = draw, below = away win. The full scoreline matrix lets the Monte Carlo simulator sample a realistic (2–1, 0–0, 3–2...) for each of the tournament's 103 matches per simulation.",
  },
];

const DISCOVERIES = [
  [
    {
      tag: "THE LEAKAGE BUG",
      tagColor: "text-red-500",
      title: "Found a Dixon-Coles leak in the original pipeline",
      desc: "The DC features (dc_home_win_prob, dc_lambda, dc_mu) used as model inputs were computed from a Dixon-Coles model fitted on all data through 2026-03-31 — the same window we were validating on. Refitting DC on pre-holdout data only changed dc_home_win_prob by >2% for 629 of the 748 holdout matches. The honest validation log loss is 0.804, not the inflated number we'd have reported.",
    },
    {
      tag: "MORE DATA, NOT CLEANER",
      tagColor: "text-amber-500",
      title: "Aggressive filtering hurts every metric",
      desc: "I tested four filter strategies: drop friendlies (already filtered), drop weak teams (ELO < 1500), modern era only (2020+), and all three combined. Every aggressive filter HURT performance on tournament matches. Even Andorra-Liechtenstein qualifiers teach the model real things about Poisson goal distributions, home advantage, and underdog conversion rates.",
    },
  ],
  [
    {
      tag: "THE DRAW WALL",
      tagColor: "text-cyan",
      title: "Football's irreducible structural limit",
      desc: "Football has ~22% draws but every calibrated model predicts draws as the modal outcome only 3-9% of the time. Tested 5 architectures (XGB classifier, hybrid Poisson, ordinal XGB, mord proportional-odds, TabNet). Every approach that solved draw recall blew up the log loss. The tradeoff is structural to the sport, not specific to any one model.",
    },
    {
      tag: "STATSBOMB CARRIES SIGNAL",
      tagColor: "text-green-500",
      title: "Intl tournament data > generic club stats",
      desc: "The new B5 features (international xG, intl form, chemistry) make up 16% of the feature count but 26% of model attention. They help most on tournament matches specifically — +0.6% accuracy improvement on continental finals and World Cup matches, where their training distribution matches the prediction context.",
    },
  ],
  [
    {
      tag: "WHY POISSON BEATS CLASSIFIER",
      tagColor: "text-purple",
      title: "A draw is not a positive event — it's the absence of a decisive one",
      desc: "The 3-way W/D/L classifier had to learn 'what makes a draw' from features. But draws aren't caused by features — they happen when two attacking outputs cancel out. The hybrid model predicts each team's expected goals separately, then derives W/D/L. Draws emerge naturally when λₕ ≈ λₐ.",
    },
    {
      tag: "LESS OVERCONFIDENT",
      tagColor: "text-pink",
      title: "No team above 14% to win the World Cup",
      desc: "The hybrid model puts Spain at 13.7% — still the favorite, but honestly hedged. No team in modern WC history has won pre-tournament at much above 18%. The model spreads probability evenly across the top 10 contenders, which is what a well-calibrated tournament forecast should do — not concentrate all certainty on one or two teams.",
    },
  ],
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
          A hybrid Poisson goal-scoring model trained on 6,162 modern-era
          international matches, combining team strength ratings, squad
          quality, recent form, and StatsBomb international tournament data
          to simulate the entire tournament 100,000 times.
        </p>
      </section>

      {/* The Three Signals */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          THE THREE SIGNALS
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[600px] mb-6 md:mb-8">
          The model combines three independent sources of team strength. Each
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
          The hybrid architecture predicts each team&apos;s expected goals
          directly with two <Tip term="Poisson">Poisson</Tip>-loss XGBoost
          regressors, then converts those into match outcomes via the{" "}
          <Tip term="Dixon-Coles">Dixon-Coles</Tip> scoreline matrix. This
          is the same approach used by FiveThirtyEight&apos;s SPI and most
          serious football models.
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

      {/* From Expected Goals to Match Outcomes */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 bg-[#0D0D0D] border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          FROM EXPECTED GOALS TO MATCH OUTCOMES
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-7">
          Instead of predicting win/draw/loss directly, the model predicts each
          team&apos;s expected goals and lets the scoreline distribution decide.
          This is why draws emerge naturally when teams are evenly matched.
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

      {/* What I Discovered */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          WHAT I DISCOVERED
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[600px] mb-6 md:mb-8">
          Six experiments. Five architectures. One real leakage bug. Here&apos;s
          what came out of it.
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
          Validated on 748 international matches from June 2025 to March 2026 —
          the 12 months immediately before the World Cup. The model never saw
          these matches during training, and no WC 2026 match enters either
          training or validation.
        </p>

        {/* Stat cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6 md:mb-8">
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-purple">
              0.027
            </span>
            <span className="text-[13px] text-secondary">
              <Tip term="ECE">Expected Calibration Error</Tip>
            </span>
            <span className="text-xs text-[#A1A1AA]">
              When the model says 70%, it&apos;s right ~70% of the time
            </span>
          </div>
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-cyan">
              62.6%
            </span>
            <span className="text-[13px] text-secondary">Outcome Accuracy</span>
            <span className="text-xs text-[#A1A1AA]">
              vs 53% baseline (always pick higher-rated squad)
            </span>
          </div>
          <div className="flex flex-col items-center gap-2 bg-[#111111] border border-[#1A1A1A] rounded-xl p-5 md:p-6">
            <span className="font-[family-name:var(--font-anton)] text-[32px] md:text-[40px] text-pink">
              0.804
            </span>
            <span className="text-[13px] text-secondary">
              <Tip term="Log Loss">Log Loss</Tip>
            </span>
            <span className="text-xs text-[#A1A1AA]">
              Lower is better &mdash; measures prediction confidence
            </span>
          </div>
        </div>

        {/* Calibration images */}
        <div className="flex flex-col gap-5">
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/phase3_calibration.png"
              alt="Calibration curves showing predicted vs actual probabilities for the hybrid model"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
          <div className="rounded-lg overflow-hidden">
            <Image
              src="/graphs/phase3_confusion.png"
              alt="Confusion matrix showing per-class accuracy"
              width={1200}
              height={420}
              className="w-full"
            />
          </div>
        </div>
      </section>

      {/* Where the model is right and wrong */}
      <section className="px-4 md:px-12 lg:px-20 py-8 md:py-12 border-t border-border">
        <h2 className="font-[family-name:var(--font-anton)] text-[22px] md:text-[28px] tracking-wide mb-2">
          WHERE THE MODEL IS STRONG AND WEAK
        </h2>
        <p className="text-secondary text-xs md:text-sm leading-relaxed max-w-[650px] mb-6 md:mb-7">
          Error sliced by <Tip term="ELO">ELO</Tip> difference between the two
          teams. Blowouts are easy. Lopsided matches are mostly predictable.
          But close matches collapse to near-random &mdash; this is football&apos;s
          irreducible uncertainty.
        </p>

        <div className="rounded-lg overflow-hidden mb-6 md:mb-8">
          <Image
            src="/graphs/phase3_error_by_elo.png"
            alt="Error by |ELO diff| bucket — log loss and accuracy across close/moderate/clear/lopsided/blowout matches"
            width={1200}
            height={420}
            className="w-full"
          />
        </div>

        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 flex flex-col gap-2 bg-[#111111] border border-green-500/40 rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-green-500">
              BLOWOUTS &mdash; 91% ACCURACY
            </span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              Matches with &gt;400 ELO difference (e.g. Spain vs San Marino).
              The favorite wins almost every time and the model knows it.
            </p>
          </div>
          <div className="flex-1 flex flex-col gap-2 bg-[#111111] border border-cyan/40 rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-cyan">
              LOPSIDED &mdash; 70% ACCURACY
            </span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              ELO difference 200&ndash;400. Most knockout R32 and R16 matches
              fall here &mdash; the model picks the right team 7 times out of 10.
            </p>
          </div>
          <div className="flex-1 flex flex-col gap-2 bg-[#111111] border border-red-500/40 rounded-xl p-5 md:p-6">
            <span className="font-mono text-xs font-semibold tracking-[2px] text-red-500">
              CLOSE &mdash; 31% ACCURACY
            </span>
            <p className="text-[13px] text-[#A1A1AA] leading-relaxed">
              ELO difference &lt;50 (e.g. Spain vs France). Football&apos;s
              irreducible noise &mdash; even the best models collapse to
              near-random here. These are also the matches most likely to be
              draws, where calibration matters more than picking the winner.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
