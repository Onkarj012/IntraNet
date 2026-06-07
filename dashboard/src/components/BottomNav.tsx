"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { href: "/",        label: "Picks",   d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" },
  { href: "/paper",   label: "Paper",   d: "M4 7h16M4 12h16M4 17h10" },
  { href: "/futures", label: "Futures", d: "M4 19V5m0 14h16M8 16l3-4 3 2 4-6" },
  { href: "/equity",  label: "Equity",  d: "M3 3h7v7H3zM14 3h7v4h-7zM14 12h7v6h-7zM3 14h7v4H3z" },
  { href: "/ops",     label: "Ops",     d: "M12 9a3 3 0 100 6 3 3 0 000-6zM4 12h2M18 12h2M12 4v2M12 18v2" },
];

export default function BottomNav() {
  const path = usePathname();
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-50 border-t border-hair bg-bg pb-[env(safe-area-inset-bottom)] sm:hidden"
    >
      <ul className="m-0 grid list-none grid-cols-5 p-0">
        {tabs.map((t) => {
          const active = t.href === "/" ? path === "/" : path === t.href || path.startsWith(`${t.href}/`);
          return (
            <li key={t.href}>
              <Link
                href={t.href}
                className={`flex h-[60px] flex-col items-center justify-center gap-1 text-[10px] font-medium no-underline ${
                  active ? "text-accent" : "text-muted"
                }`}
              >
                <svg width={20} height={20} viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
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
