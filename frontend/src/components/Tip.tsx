"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";

/* ── Term dictionary ── */

const TERMS: Record<string, string> = {
  // General football
  "GD": "Goal Difference — goals scored minus goals conceded",
  "PTS": "Points — 3 for a win, 1 for a draw, 0 for a loss",
  "MP": "Matches Played",
  "W": "Wins",
  "D": "Draws",
  "L": "Losses",
  "caps": "Number of international matches played for their country",
  "PEN": "Match decided by penalty shootout after a draw",

  // Positions
  "GK": "Goalkeeper — the player who guards the net",
  "DF": "Defender — players who protect the goal and stop attacks",
  "MF": "Midfielder — players who link defense and attack, control the game",
  "FW": "Forward — attacking players who score goals",
  "Pos": "Playing position on the field (GK, DF, MF, FW)",

  // EA FC ratings
  "EA FC": "Player ratings from the EA Sports FC video game — each player is rated 1–99 by professional scouts",
  "OVR": "Overall Rating — a player's combined ability score (1–99) from EA FC",
  "POT": "Potential — the highest overall rating a player could reach",
  "PAC": "Pace — how fast the player can run",
  "SHO": "Shooting — accuracy and power of shots on goal",
  "PAS": "Passing — ability to deliver the ball to teammates",
  "DRI": "Dribbling — ball control and ability to beat defenders",
  "DEF": "Defending — ability to tackle and intercept the ball",
  "PHY": "Physical — strength, stamina, and aggression",
  "DIV": "Diving — goalkeeper's ability to dive and save shots",
  "HAN": "Handling — goalkeeper's ability to catch and hold the ball",
  "KIC": "Kicking — goalkeeper's distribution accuracy",
  "REF": "Reflexes — goalkeeper's reaction speed to shots",
  "REP": "Reputation — how well-known the player is internationally (1–5 stars)",
  "VAL": "Transfer Value — estimated market value of the player",

  // Model & stats
  "ELO": "A rating system borrowed from chess — teams gain points for wins and lose points for losses, weighted by opponent strength",
  "Monte Carlo": "Running the tournament thousands of times with randomized outcomes to calculate probabilities",
  "XGBoost": "A machine learning algorithm that builds decision trees to find patterns in data",
  "Random Forest": "An ML algorithm using hundreds of independent decision trees that vote on the outcome",
  "Log Loss": "Measures how confident and correct predictions are — lower is better",
  "ECE": "Expected Calibration Error — measures if predicted probabilities match reality (0 = perfect)",
  "calibration": "When a model says '70% chance', does it actually happen 70% of the time? That's calibration",
  "Poisson": "A statistical method for modeling random events like goals — used to generate realistic scorelines",
  "ensemble": "Multiple ML models combined — they vote on each prediction, reducing individual errors",
  "feature": "A measurable input the model uses to make predictions (e.g., ELO rating, recent form)",
  "form": "A team's recent performance — win rate, goals scored/conceded over the last few matches",
  "head-to-head": "Historical record between two specific teams",
  "confederation": "Regional football governing body (UEFA = Europe, CONMEBOL = South America, etc.)",
  "CONCACAF": "North/Central America & Caribbean football confederation",
  "AFC": "Asian Football Confederation",
  "UEFA": "Union of European Football Associations",
  "CONMEBOL": "South American football confederation",
  "CAF": "Confederation of African Football",
};

/* ── Component ── */

interface TipProps {
  term: string;
  desc?: string;
  children?: React.ReactNode;
}

export default function Tip({ term, desc, children }: TipProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number; above: boolean }>({ top: 0, left: 0, above: false });
  const ref = useRef<HTMLSpanElement>(null);
  const timeout = useRef<ReturnType<typeof setTimeout>>();
  const [mounted, setMounted] = useState(false);

  const text = desc || TERMS[term] || term;

  useEffect(() => { setMounted(true); }, []);

  const show = useCallback(() => {
    clearTimeout(timeout.current);
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      const above = rect.top > 160;
      setPos({
        top: above ? rect.top : rect.bottom,
        left: Math.min(Math.max(rect.left + rect.width / 2, 130), window.innerWidth - 130),
        above,
      });
    }
    setOpen(true);
  }, []);

  const hide = useCallback(() => {
    timeout.current = setTimeout(() => setOpen(false), 150);
  }, []);

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, { passive: true });
    return () => window.removeEventListener("scroll", close);
  }, [open]);

  const tooltip = open && mounted
    ? createPortal(
        <div
          className={`fixed z-[9999] w-[220px] md:w-[260px] px-3 py-2.5 bg-[#1A1A1A] border border-border rounded-lg text-[11px] md:text-xs text-[#A1A1AA] leading-relaxed shadow-lg pointer-events-none`}
          style={{
            top: pos.above ? pos.top - 8 : pos.top + 8,
            left: pos.left,
            transform: pos.above
              ? "translate(-50%, -100%)"
              : "translate(-50%, 0)",
          }}
        >
          <span className="text-foreground font-semibold">{term}</span>
          <br />
          {text}
        </div>,
        document.body
      )
    : null;

  return (
    <span
      ref={ref}
      className="relative inline-flex items-center cursor-help"
      onMouseEnter={show}
      onMouseLeave={hide}
      onTouchStart={show}
    >
      <span className="border-b border-dotted border-secondary/50 hover:border-purple transition-colors">
        {children ?? term}
      </span>
      {tooltip}
    </span>
  );
}
