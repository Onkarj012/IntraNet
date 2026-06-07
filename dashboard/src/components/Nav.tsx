"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/",        label: "Picks"   },
  { href: "/paper",   label: "Paper"   },
  { href: "/futures", label: "Futures" },
  { href: "/equity",  label: "Equity"  },
  { href: "/ops",     label: "Ops"     },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-50 border-b border-hair bg-bg">
      <div className="mx-auto flex h-[60px] max-w-[1200px] items-center justify-between gap-5 px-5">
        <Link href="/" className="flex items-center gap-2.5 no-underline">
          <span className="grid h-7 w-7 shrink-0 place-items-center rounded-[4px] bg-accent text-[13px] font-bold text-white">
            O
          </span>
          <span className="text-[15px] font-semibold tracking-[-0.02em] text-ink">OptiNet</span>
        </Link>

        <nav className="hidden items-center gap-8 sm:flex">
          {links.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`border-b pb-0.5 text-[14px] font-medium no-underline transition-colors duration-150 ${
                  active
                    ? "border-accent text-ink"
                    : "border-transparent text-muted hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-2">
          <span
            className="h-1.5 w-1.5 rounded-full bg-up"
            style={{ animation: "dot-pulse 2s ease infinite" }}
          />
        </div>
      </div>
    </header>
  );
}
