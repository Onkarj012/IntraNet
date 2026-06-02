"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const I = {
  recos: "M12 3l2.4 5 5.6.8-4 3.9 1 5.5L12 16.5 6.9 18l1-5.5-4-3.9 5.6-.8z",
  paper: "M4 7h16M4 12h16M4 17h10",
  futures: "M4 19V5M4 19h16M8 16l3-4 3 2 4-6",
  equity: "M4 5h7v7H4zM13 5h7v4h-7zM13 13h7v6h-7zM4 15h7v4H4z",
  ops: "M12 9a3 3 0 100 6 3 3 0 000-6zM4 12h2M18 12h2M12 4v2M12 18v2",
};

const tabs: { href: string; label: string; d: string }[] = [
  { href: "/", label: "Recos", d: I.recos },
  { href: "/paper", label: "Paper", d: I.paper },
  { href: "/futures", label: "Futures", d: I.futures },
  { href: "/equity", label: "Equity", d: I.equity },
  { href: "/ops", label: "Ops", d: I.ops },
];

export default function BottomNav() {
  const pathname = usePathname();
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-50 border-t border-hair bg-bg/90 backdrop-blur-xl md:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <ul className="grid grid-cols-5">
        {tabs.map((t) => {
          const active = t.href === "/" ? pathname === "/" : pathname.startsWith(t.href);
          return (
            <li key={t.href}>
              <Link
                href={t.href}
                className={`flex flex-col items-center gap-1 py-2.5 text-[10px] font-medium ${
                  active ? "text-accent" : "text-ink-60"
                }`}
              >
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d={t.d} />
                </svg>
                {t.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
