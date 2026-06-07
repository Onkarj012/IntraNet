'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const tabs = [
  { href: '/', label: 'Brief', icon: '◈' },
  { href: '/ledger', label: 'Ledger', icon: '▤' },
  { href: '/performance', label: 'Perf', icon: '◎' },
  { href: '/ops', label: 'Ops', icon: '⚙' },
]

export default function MobileNav() {
  const pathname = usePathname()
  return (
    <nav style={{
      position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 50,
      height: 56, background: 'var(--bg-page)',
      borderTop: 'var(--border-solid)',
      display: 'flex',
    }}>
      {tabs.map(({ href, label, icon }) => {
        const active = pathname === href
        return (
          <Link key={href} href={href} style={{
            flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
            justifyContent: 'center', gap: 3, textDecoration: 'none',
            color: active ? 'var(--accent)' : 'var(--text-secondary)',
            borderTop: active ? '2px solid var(--accent)' : '2px solid transparent',
            fontSize: 18, transition: 'color 0.15s',
          }}>
            <span>{icon}</span>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-ui)' }}>{label}</span>
          </Link>
        )
      })}
    </nav>
  )
}
