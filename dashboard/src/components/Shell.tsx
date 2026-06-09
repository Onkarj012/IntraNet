'use client'
import { useEffect, useState } from 'react'
import Navbar from './Navbar'
import MobileNav from './MobileNav'
import HaltBanner from './HaltBanner'
import { useIsMobile } from '@/hooks/useIsMobile'

export default function Shell({ children }: { children: React.ReactNode }) {
  const [haltActive, setHaltActive] = useState(false)
  const isMobile = useIsMobile()

  useEffect(() => {
    fetch('/api/drift').then(r => r.json()).then(d => setHaltActive(!!d.halt_file_exists))
  }, [])

  return (
    <>
      {!isMobile && <Navbar />}
      {haltActive && (
        <div style={{
          position: 'sticky', top: isMobile ? 0 : 56, zIndex: 49,
          background: 'var(--halt)', borderBottom: '1px solid rgba(252,165,165,0.2)',
          padding: '10px 16px', fontSize: 13, color: '#fca5a5', textAlign: 'center',
        }}>
          ⚠ System halted — kill-switch active. All entries blocked.
        </div>
      )}
      <main style={{
        minHeight: '100dvh',
        background: 'var(--bg-page)',
        paddingBottom: isMobile ? 56 : 0,
      }}>
        {children}
      </main>
      {isMobile && <MobileNav />}
    </>
  )
}
