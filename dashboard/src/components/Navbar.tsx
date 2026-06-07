'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'

const navLinks = [
  { href: '/', label: 'Brief' },
  { href: '/ledger', label: 'Ledger' },
  { href: '/performance', label: 'Performance' },
  { href: '/ops', label: 'Ops' },
]

function ISTClock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const fmt = new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      day: '2-digit', month: 'short', year: 'numeric',
      hour12: false,
    })
    const tick = () => setTime(fmt.format(new Date()))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <span className="mono text-secondary" style={{ fontSize: 13 }}>{time} IST</span>
  )
}

export default function Navbar() {
  const pathname = usePathname()
  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 50, height: 56,
      background: 'var(--bg-page)',
      borderBottom: 'var(--border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 32px',
    }}>
      <span className="mono" style={{ fontSize: 16, color: 'var(--accent)', fontWeight: 500 }}>
        StockXpert
      </span>
      <div style={{ display: 'flex', gap: 32 }}>
        {navLinks.map(({ href, label }) => {
          const active = pathname === href
          return (
            <Link key={href} href={href} style={{
              fontSize: 14, textDecoration: 'none',
              color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
              borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
              paddingBottom: 4,
              transition: 'color 0.15s',
            }}
            onMouseEnter={e => { if (!active) (e.target as HTMLElement).style.color = 'var(--text-primary)' }}
            onMouseLeave={e => { if (!active) (e.target as HTMLElement).style.color = 'var(--text-secondary)' }}
            >
              {label}
            </Link>
          )
        })}
      </div>
      <ISTClock />
    </nav>
  )
}
