"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/live", label: "Live", indicator: true },
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
          <Image
            src="/wc2026-logo.jpeg"
            alt="FIFA World Cup 2026"
            width={22}
            height={34}
            className="h-8 w-auto"
            priority
          />
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
              className={`flex items-center gap-1.5 text-sm transition-colors hover:text-purple ${
                pathname === link.href ? "text-purple" : "text-secondary"
              }`}
            >
              {link.indicator && (
                <span className="w-1.5 h-1.5 rounded-full bg-pink animate-livePulse" />
              )}
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
              className={`flex items-center gap-2 px-4 py-3 text-sm border-b border-border transition-colors ${
                pathname === link.href ? "text-purple bg-purple/5" : "text-secondary"
              }`}
            >
              {link.indicator && (
                <span className="w-1.5 h-1.5 rounded-full bg-pink animate-livePulse" />
              )}
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
