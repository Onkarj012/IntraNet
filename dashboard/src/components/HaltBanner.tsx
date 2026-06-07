'use client'
export default function HaltBanner({ visible }: { visible: boolean }) {
  if (!visible) return null
  return (
    <div style={{
      position: 'sticky', top: 56, zIndex: 49,
      background: 'var(--halt)', borderBottom: '1px solid rgba(252,165,165,0.2)',
      padding: '10px 32px', fontSize: 13, color: '#fca5a5',
      textAlign: 'center',
    }}>
      ⚠ System halted — kill-switch active. All entries blocked.
    </div>
  )
}
