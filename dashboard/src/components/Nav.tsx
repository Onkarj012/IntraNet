"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Recommendations", short: "Recos" },
  { href: "/paper", label: "Paper Trading", short: "Paper" },
  { href: "/futures", label: "Futures", short: "Futures" },
  { href: "/equity", label: "Equity", short: "Equity" },
  { href: "/ops", label: "Operations", short: "Ops" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-50 border-b border-hair bg-bg/70 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-[1180px] items-center justify-between px-5 sm:px-6">
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-input bg-accent text-[16px] font-black text-white shadow-[0_0_24px_rgba(110,104,238,0.5)]">
            O
          </span>
          <span className="text-[16px] font-bold tracking-[-0.02em] text-ink">
            OptiNet
          </span>
        </Link>

        <nav className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-1 rounded-pill border border-hair bg-card/60 p-1 backdrop-blur-md md:flex">
          {links.map((l) => {
            const active =
              l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-pill px-4 py-1.5 text-[13px] font-medium ${
                  active ? "bg-raised text-ink" : "text-ink-60 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-[12px] text-ink-60">
            <span className="h-1.5 w-1.5 rounded-pill bg-up animate-pulse" />
            <span className="hidden sm:inline">paper trading </span>live
          </span>
        </div>
      </div>
    </header>
  );
}
