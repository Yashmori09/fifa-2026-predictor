"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/tournament", label: "Tournament" },
  { href: "/predict", label: "Match Predictor" },
  { href: "/squads", label: "Squads" },
  { href: "/methodology", label: "Methodology" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-background border-b border-border">
      <div className="flex items-center justify-between px-4 md:px-12 h-14 md:h-16">
        <Link href="/" className="flex items-center gap-[10px]">
          <svg
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-yellow-400"
          >
            <path d="M10 14.66v1.626a2 2 0 0 1-.976 1.696A5 5 0 0 0 7 21.978" />
            <path d="M14 14.66v1.626a2 2 0 0 0 .976 1.696A5 5 0 0 1 17 21.978" />
            <path d="M18 9h1.5a1 1 0 0 0 0-5H18" />
            <path d="M4 22h16" />
            <path d="M6 9a6 6 0 0 0 12 0V3a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1z" />
            <path d="M6 9H4.5a1 1 0 0 1 0-5H6" />
          </svg>
          <span className="text-[18px] md:text-[20px] font-[family-name:var(--font-anton)] tracking-[1px]">
            FIFA 2026
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm transition-colors hover:text-purple ${
                pathname === link.href ? "text-purple" : "text-secondary"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setOpen(!open)}
          className="md:hidden flex flex-col gap-1.5 p-2"
          aria-label="Toggle menu"
        >
          <span className={`block w-5 h-0.5 bg-foreground transition-transform ${open ? "rotate-45 translate-y-[4px]" : ""}`} />
          <span className={`block w-5 h-0.5 bg-foreground transition-opacity ${open ? "opacity-0" : ""}`} />
          <span className={`block w-5 h-0.5 bg-foreground transition-transform ${open ? "-rotate-45 -translate-y-[4px]" : ""}`} />
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden flex flex-col border-t border-border bg-background">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setOpen(false)}
              className={`px-4 py-3 text-sm border-b border-border transition-colors ${
                pathname === link.href ? "text-purple bg-purple/5" : "text-secondary"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
