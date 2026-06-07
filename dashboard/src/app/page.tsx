'use client'
import { useEffect, useState } from 'react'
import { useIsMobile } from '@/hooks/useIsMobile'

interface Driver { momentum: number; reversal: number; breakout: number; sentiment: number; macro: number }
interface Pick {
  symbol: string; direction: string; p_up: number; entry: number; target: number;
  stop: number; rr: number; regime: string; expected_value: number; drivers: Driver
}
interface Futures {
  tradeable: boolean; entry: number | null; target: number | null; stop: number | null;
  roi: number | null; rr: number | null; lot: number | null; confidence: number | null;
  vix: number; stance: string | null; note: string | null
}
interface Recs { generated_at: string; equity: { exposure: number; picks: Pick[] }; futures: Futures }

function regimePillClass(regime: string) {
  if (regime.includes('trending')) return 'pill pill-regime-trending'
  if (regime.includes('choppy')) return 'pill pill-regime-choppy'
  if (regime.includes('crisis')) return 'pill pill-regime-crisis'
  return 'pill pill-regime-neutral'
}
function vixColor(vix: number) { return vix < 15 ? 'var(--win)' : vix <= 22 ? 'var(--warn)' : 'var(--loss)' }
function driverIcon(score: number) { return score > 0.6 ? '▲' : score < 0.4 ? '▼' : '○' }

function Bar({ pct, color = 'var(--accent)' }: { pct: number; color?: string }) {
  return (
    <div style={{ height: 4, borderRadius: 2, background: '#1a1a1a', flex: 1 }}>
      <div style={{ height: '100%', borderRadius: 2, background: color, width: `${pct * 100}%` }} />
    </div>
  )
}

// ─── DESKTOP COMPONENTS ────────────────────────────────────────────────────

function PickCard({ pick, capital }: { pick: Pick; capital: number }) {
  const [expanded, setExpanded] = useState(false)
  const drivers = Object.entries(pick.drivers) as [string, number][]
  return (
    <div className="card" style={{ cursor: 'pointer' }} onClick={() => setExpanded(e => !e)}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <span className="mono" style={{ fontSize: 18, fontWeight: 500 }}>{pick.symbol}</span>
        <span className="pill pill-win" style={{ margin: '0 8px' }}>LONG ↑</span>
        <span className="mono" style={{ fontSize: 18, fontWeight: 500 }}>₹{capital.toLocaleString('en-IN')}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 16 }}>
        {[['Entry', `₹${pick.entry.toLocaleString('en-IN')}`, 'var(--text-primary)'],
          ['Stop', `₹${pick.stop.toLocaleString('en-IN')}`, 'var(--loss)'],
          ['Target', `₹${pick.target.toLocaleString('en-IN')}`, 'var(--win)'],
          ['R:R', `${pick.rr}×`, 'var(--text-primary)']].map(([label, val, color]) => (
          <div key={label}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
            <div className="mono" style={{ fontSize: 14, color }}>{val}</div>
          </div>
        ))}
      </div>
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)', width: 72 }}>Conviction</span>
          <Bar pct={pick.p_up} />
          <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', width: 36, textAlign: 'right' }}>{Math.round(pick.p_up * 100)}%</span>
        </div>
      </div>
      {!expanded ? (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {drivers.map(([name, score]) => (
            <span key={name} className="pill pill-regime-neutral" style={{ fontSize: 11 }}>{name} {driverIcon(score)}</span>
          ))}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, borderTop: 'var(--border)', paddingTop: 14 }}>
          {drivers.map(([name, score]) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', width: 80, textTransform: 'capitalize' }}>{name}</span>
              <div style={{ width: 120 }}><Bar pct={score} color={score > 0.6 ? 'var(--win)' : score < 0.4 ? 'var(--loss)' : 'var(--warn)'} /></div>
              <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', width: 36 }}>{score.toFixed(2)}</span>
              <span style={{ fontSize: 12 }}>{driverIcon(score)}</span>
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 14, paddingTop: 12, borderTop: 'var(--border)' }}>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{pick.regime.replace(/_/g, ' ')} · {pick.regime.includes('trending') ? 100 : 80}% exposure</span>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{expanded ? '▲ drivers' : '▼ drivers'}</span>
      </div>
    </div>
  )
}

function CapitalStrip({ picks, exposure }: { picks: Pick[]; exposure: number }) {
  const totalCapital = 100000
  const effectiveCapital = totalCapital * (exposure / 100)
  const nPicks = picks.length
  const perPick = nPicks > 0 ? Math.round(effectiveCapital / nPicks) : 0
  const items = [
    { label: 'Total capital', val: `₹${(totalCapital / 100000).toFixed(1)}L` },
    { label: 'Regime exposure', val: `${exposure}%` },
    { label: 'Effective capital', val: `₹${(effectiveCapital / 100000).toFixed(1)}L` },
    { label: 'Picks today', val: `${nPicks}` },
    { label: 'Per pick', val: `₹${perPick.toLocaleString('en-IN')}`, accent: true },
  ]
  return (
    <div style={{ display: 'flex', gap: 12, padding: '16px 32px' }}>
      {items.map(({ label, val, accent }) => (
        <div key={label} className="metric-card" style={{ flex: 1, borderLeft: accent ? '2px solid var(--accent)' : undefined }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
          <div className="mono" style={{ fontSize: 20, fontWeight: 500 }}>{val}</div>
        </div>
      ))}
    </div>
  )
}

function FuturesCard({ futures }: { futures: Futures }) {
  return (
    <div className="card" style={{ maxWidth: 600, margin: '0 32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span className="mono" style={{ fontSize: 16, fontWeight: 500 }}>NIFTY 50 Futures</span>
        {futures.tradeable
          ? <span className="pill pill-live">LIVE <span className="pulse">●</span></span>
          : <span className="pill pill-halt">STAND ASIDE</span>}
      </div>
      {!futures.tradeable ? (
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 16 }}>⊘&nbsp; {futures.note}</div>
          <div style={{ display: 'flex', gap: 24 }}>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>VIX · {futures.vix}</span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Confidence · –</span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Lots · –</span>
          </div>
        </div>
      ) : (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
            {[['Entry', futures.entry?.toLocaleString('en-IN') ?? '–', 'var(--text-primary)'],
              ['Target', futures.target?.toLocaleString('en-IN') ?? '–', 'var(--win)'],
              ['Stop', futures.stop?.toLocaleString('en-IN') ?? '–', 'var(--loss)'],
              ['Lots', `${futures.lot ?? '–'}`, 'var(--text-primary)']].map(([label, val, color]) => (
              <div key={label}>
                <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
                <div className="mono" style={{ fontSize: 14, color }}>{val}</div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 24, borderTop: 'var(--border)', paddingTop: 14 }}>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>ROI · {futures.roi}%</span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>R:R · {futures.rr}×</span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Confidence · {futures.confidence}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── MOBILE COMPONENTS ─────────────────────────────────────────────────────

function MobilePickCard({ pick, capital }: { pick: Pick; capital: number }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div style={{
      background: 'var(--bg-surface)', borderRadius: 12,
      border: '1px dashed rgba(255,255,255,0.1)',
      overflow: 'hidden',
    }}>
      {/* Top bar: symbol + direction + capital */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 16px 10px' }}>
        <span className="mono" style={{ fontSize: 17, fontWeight: 500 }}>{pick.symbol}</span>
        <span className="pill pill-win" style={{ fontSize: 10 }}>LONG ↑</span>
        <span className="mono" style={{ fontSize: 15, fontWeight: 500, color: 'var(--accent)' }}>₹{capital.toLocaleString('en-IN')}</span>
      </div>
      {/* Entry / Stop / Target in a row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 0, padding: '0 16px 14px' }}>
        {[['Entry', `₹${pick.entry.toLocaleString('en-IN')}`, 'var(--text-primary)'],
          ['Stop', `₹${pick.stop.toLocaleString('en-IN')}`, 'var(--loss)'],
          ['Target', `₹${pick.target.toLocaleString('en-IN')}`, 'var(--win)'],
          ['R:R', `${pick.rr}×`, 'var(--text-primary)']].map(([label, val, color]) => (
          <div key={label}>
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginBottom: 3 }}>{label}</div>
            <div className="mono" style={{ fontSize: 13, color }}>{val}</div>
          </div>
        ))}
      </div>
      {/* Conviction bar */}
      <div style={{ padding: '0 16px 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Bar pct={pick.p_up} />
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>{Math.round(pick.p_up * 100)}% conviction</span>
      </div>
      {/* Expand drivers */}
      <button onClick={() => setExpanded(e => !e)} style={{
        width: '100%', padding: '10px 16px', background: 'rgba(255,255,255,0.02)',
        border: 'none', borderTop: 'var(--border)', color: 'var(--text-secondary)',
        fontSize: 11, cursor: 'pointer', textAlign: 'left', display: 'flex', justifyContent: 'space-between',
        fontFamily: 'var(--font-ui)',
      }}>
        <span>{pick.regime.replace(/_/g, ' ')}</span>
        <span>{expanded ? '▲ drivers' : '▼ drivers'}</span>
      </button>
      {expanded && (
        <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {(Object.entries(pick.drivers) as [string, number][]).map(([name, score]) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', width: 72, textTransform: 'capitalize' }}>{name}</span>
              <div style={{ flex: 1 }}><Bar pct={score} color={score > 0.6 ? 'var(--win)' : score < 0.4 ? 'var(--loss)' : 'var(--warn)'} /></div>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', width: 32, textAlign: 'right' }}>{score.toFixed(2)}</span>
              <span style={{ fontSize: 11, width: 14 }}>{driverIcon(score)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MobileFuturesCard({ futures }: { futures: Futures }) {
  return (
    <div style={{ background: 'var(--bg-surface)', borderRadius: 12, border: '1px dashed rgba(255,255,255,0.1)', padding: '14px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <span className="mono" style={{ fontSize: 14, fontWeight: 500 }}>NIFTY 50 Futures</span>
        {futures.tradeable
          ? <span className="pill pill-live" style={{ fontSize: 10 }}>LIVE <span className="pulse">●</span></span>
          : <span className="pill pill-halt" style={{ fontSize: 10 }}>STAND ASIDE</span>}
      </div>
      {!futures.tradeable ? (
        <div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 10 }}>⊘ {futures.note}</div>
          <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>VIX · {futures.vix}</span>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[['Entry', futures.entry?.toLocaleString('en-IN') ?? '–', 'var(--text-primary)'],
            ['Target', futures.target?.toLocaleString('en-IN') ?? '–', 'var(--win)'],
            ['Stop', futures.stop?.toLocaleString('en-IN') ?? '–', 'var(--loss)'],
            ['Lots', `${futures.lot ?? '–'}`, 'var(--text-primary)'],
            ['R:R', `${futures.rr ?? '–'}×`, 'var(--text-primary)'],
            ['Confidence', `${futures.confidence ?? '–'}%`, 'var(--text-primary)']].map(([label, val, color]) => (
            <div key={label}>
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginBottom: 3 }}>{label}</div>
              <div className="mono" style={{ fontSize: 14, color }}>{val}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── PAGE ──────────────────────────────────────────────────────────────────

export default function BriefPage() {
  const [data, setData] = useState<Recs | null>(null)
  const isMobile = useIsMobile()
  useEffect(() => { fetch('/api/recommendations').then(r => r.json()).then(setData) }, [])
  if (!data) return <div style={{ padding: 32, color: 'var(--text-secondary)' }}>Loading…</div>

  const { equity, futures } = data
  const picks = equity.picks
  const nPicks = picks.length
  const regime = picks[0]?.regime ?? 'neutral'
  const perPick = nPicks > 0 ? Math.round((100000 * equity.exposure / 100) / nPicks) : 0
  const updatedAt = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(new Date(data.generated_at))

  // ── MOBILE ──
  if (isMobile) {
    const effectiveCapital = 100000 * (equity.exposure / 100)
    return (
      <div style={{ padding: '0 0 8px' }}>
        {/* Compact header bar */}
        <div style={{ padding: '12px 16px', display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', borderBottom: 'var(--border)', background: 'var(--bg-surface)' }}>
          <span className={regimePillClass(regime)} style={{ fontSize: 10 }}>{regime.replace(/_/g, ' ')}</span>
          <span className="pill pill-regime-neutral mono" style={{ fontSize: 10 }}>VIX · <span style={{ color: vixColor(futures.vix) }}>{futures.vix}</span></span>
          <span className="pill pill-regime-neutral" style={{ fontSize: 10 }}>{nPicks} picks</span>
          <span className="mono" style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-secondary)' }}>{updatedAt} IST</span>
        </div>

        {/* Capital summary strip — 2 items per row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'rgba(255,255,255,0.05)', margin: '0 0 16px' }}>
          {[
            ['Exposure', `${equity.exposure}%`],
            ['Effective capital', `₹${(effectiveCapital / 100000).toFixed(1)}L`],
            ['Picks', `${nPicks}`],
            ['Per pick', `₹${perPick.toLocaleString('en-IN')}`],
          ].map(([label, val]) => (
            <div key={label} style={{ background: 'var(--bg-surface)', padding: '12px 14px' }}>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>{label}</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 500 }}>{val}</div>
            </div>
          ))}
        </div>

        {/* Pick cards stacked */}
        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Equity picks
          </div>
          {picks.length === 0
            ? <p className="mono" style={{ color: 'var(--text-muted)', fontSize: 13 }}>No picks generated</p>
            : picks.map(p => <MobilePickCard key={p.symbol} pick={p} capital={perPick} />)
          }

          {/* Futures */}
          <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', marginTop: 8, marginBottom: 4 }}>
            NIFTY futures
          </div>
          <MobileFuturesCard futures={futures} />
        </div>
      </div>
    )
  }

  // ── DESKTOP ──
  return (
    <div style={{ paddingBottom: 48 }}>
      <div style={{ background: 'var(--bg-surface)', borderBottom: 'var(--border)', padding: '14px 32px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span className={regimePillClass(regime)}>{regime.replace(/_/g, ' ')}</span>
        <span className="pill pill-regime-neutral mono">Exposure · {equity.exposure}%</span>
        <span className="pill pill-regime-neutral mono" style={{ color: vixColor(futures.vix) }}>VIX · {futures.vix}</span>
        <span className="pill pill-regime-neutral">{nPicks} picks today</span>
        <span className="mono" style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-secondary)' }}>Updated {updatedAt} IST</span>
      </div>
      <div style={{ padding: '24px 32px 0' }}>
        <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 16 }}>
          Equity picks — {new Date(data.generated_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
        </h2>
        {picks.length === 0
          ? <p className="mono" style={{ color: 'var(--text-muted)', fontSize: 14 }}>No picks generated — system may be in crisis regime</p>
          : <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
              {picks.map(p => <PickCard key={p.symbol} pick={p} capital={perPick} />)}
            </div>
        }
      </div>
      <div style={{ marginTop: 24 }}><CapitalStrip picks={picks} exposure={equity.exposure} /></div>
      <hr style={{ border: 'none', borderTop: 'var(--border)', margin: '8px 32px 24px' }} />
      <div>
        <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 16, padding: '0 32px' }}>NIFTY futures</h2>
        <FuturesCard futures={futures} />
      </div>
    </div>
  )
}
