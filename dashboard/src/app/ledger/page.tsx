'use client'
import { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts'
import { useIsMobile } from '@/hooks/useIsMobile'

interface Trade {
  date: string; symbol: string; direction: string; dir_p: string;
  entry: string; stop: string; high: string; low: string; close: string;
  outcome: string; pnl_inr: string; pnl_pct: string; regime: string;
  exposure: string; capital_after: string;
}

function fmtPnl(v: number) {
  const s = Math.abs(v).toLocaleString('en-IN')
  return v >= 0 ? `+₹${s}` : `−₹${s}`
}
function pnlColor(v: number) { return v >= 0 ? 'var(--win)' : 'var(--loss)' }

function fmtLakh(v: number) {
  if (v >= 100000) return `₹${(v / 100000).toFixed(2)}L`
  return `₹${v.toLocaleString('en-IN')}`
}

function regimePill(regime: string) {
  const cls = regime.includes('trending') ? 'pill-regime-trending'
    : regime.includes('choppy') ? 'pill-regime-choppy'
    : regime.includes('crisis') ? 'pill-regime-crisis'
    : 'pill-regime-neutral'
  return <span className={`pill ${cls}`} style={{ fontSize: 10 }}>{regime.replace(/_/g, ' ')}</span>
}

function outcomePill(o: string) {
  const cls = o === 'WIN' ? 'pill-win' : o === 'LOSS' ? 'pill-loss' : 'pill-stop'
  return <span className={`pill ${cls}`}>{o}</span>
}

// Heatmap
const heatColor = (pnl: number) => {
  if (pnl > 2000) return '#166534'
  if (pnl > 1000) return '#15803d'
  if (pnl > 0)    return '#14532d'
  if (pnl > -1000) return '#7f1d1d'
  if (pnl > -2000) return '#991b1b'
  return '#450a0a'
}

function CalendarHeatmap({ trades }: { trades: Trade[] }) {
  const byDate = useMemo(() => {
    const m: Record<string, { pnl: number; count: number; symbols: string[] }> = {}
    for (const t of trades) {
      if (!m[t.date]) m[t.date] = { pnl: 0, count: 0, symbols: [] }
      m[t.date].pnl += parseFloat(t.pnl_inr)
      m[t.date].count++
      m[t.date].symbols.push(t.symbol)
    }
    return m
  }, [trades])

  const dates = Object.keys(byDate).sort()
  if (dates.length === 0) return null

  // Group by month
  const months: Record<string, Set<string>> = {}
  for (const d of dates) {
    const ym = d.slice(0, 7)
    if (!months[ym]) months[ym] = new Set()
    months[ym].add(d)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {Object.entries(months).map(([ym, daySet]) => {
        const [year, mon] = ym.split('-').map(Number)
        const monthLabel = new Date(year, mon - 1, 1).toLocaleDateString('en-IN', { month: 'short', year: 'numeric' })
        const daysInMonth = new Date(year, mon, 0).getDate()
        const firstDow = new Date(year, mon - 1, 1).getDay() // 0=Sun
        // ISO: Mon=0
        const offset = (firstDow + 6) % 7

        return (
          <div key={ym}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>{monthLabel}</div>
            <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              {/* Day headers */}
              {['M','T','W','T','F','S','S'].map((d,i) => (
                <div key={i} style={{ width: 28, height: 16, fontSize: 9, color: 'var(--text-muted)', textAlign: 'center', lineHeight: '16px', fontFamily: 'var(--font-mono)' }}>{d}</div>
              ))}
              {/* Offset cells */}
              {Array.from({ length: offset }).map((_, i) => (
                <div key={`off-${i}`} style={{ width: 28, height: 28 }} />
              ))}
              {Array.from({ length: daysInMonth }).map((_, i) => {
                const day = i + 1
                const dateStr = `${ym}-${String(day).padStart(2, '0')}`
                const info = byDate[dateStr]
                const bg = info ? heatColor(info.pnl) : 'var(--bg-surface)'
                return (
                  <div key={day}
                    title={info ? `${dateStr} | ${info.count} trades · ${fmtPnl(info.pnl)} | ${info.symbols.join(', ')}` : dateStr}
                    style={{
                      width: 28, height: 28, borderRadius: 4, background: bg,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 9, color: 'rgba(255,255,255,0.5)', fontFamily: 'var(--font-mono)',
                      border: info ? 'none' : '0.5px solid rgba(255,255,255,0.06)',
                      cursor: info ? 'pointer' : 'default',
                    }}
                  >{day}</div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function LedgerPage() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [outcomeFilter, setOutcomeFilter] = useState<string[]>([])
  const [symbolSearch, setSymbolSearch] = useState('')
  const [sortCol, setSortCol] = useState<string>('date')
  const [sortAsc, setSortAsc] = useState(false)
  const isMobile = useIsMobile()

  useEffect(() => {
    fetch('/api/ledger').then(r => r.text()).then(csv => {
      const result = Papa.parse<Trade>(csv, { header: true, skipEmptyLines: true })
      setTrades(result.data)
    })
  }, [])

  const today = new Date().toISOString().slice(0, 10)

  const stats = useMemo(() => {
    if (!trades.length) return null
    const wins = trades.filter(t => t.outcome === 'WIN').length
    const totalPnl = trades.reduce((s, t) => s + parseFloat(t.pnl_inr), 0)
    const pnlByDay: Record<string, number> = {}
    for (const t of trades) pnlByDay[t.date] = (pnlByDay[t.date] ?? 0) + parseFloat(t.pnl_inr)
    const dayPnls = Object.values(pnlByDay)
    return {
      trades: trades.length,
      winRate: ((wins / trades.length) * 100).toFixed(1),
      totalPnl,
      avgPnl: totalPnl / trades.length,
      bestDay: Math.max(...dayPnls),
      worstDay: Math.min(...dayPnls),
    }
  }, [trades])

  const equityCurve = useMemo(() => {
    const byDate: Record<string, number> = {}
    for (const t of trades) byDate[t.date] = parseFloat(t.capital_after)
    return Object.entries(byDate).sort(([a], [b]) => a.localeCompare(b)).map(([date, capital]) => ({ date, capital }))
  }, [trades])

  const filtered = useMemo(() => {
    let rows = [...trades]
    if (outcomeFilter.length) rows = rows.filter(t => outcomeFilter.includes(t.outcome))
    if (symbolSearch) rows = rows.filter(t => t.symbol.toUpperCase().includes(symbolSearch.toUpperCase()))
    rows.sort((a, b) => {
      let av: any = a[sortCol as keyof Trade], bv: any = b[sortCol as keyof Trade]
      if (['pnl_inr','entry','stop','close','capital_after'].includes(sortCol)) { av = parseFloat(av); bv = parseFloat(bv) }
      return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    })
    const todayRows = rows.filter(t => t.date === today)
    const rest = rows.filter(t => t.date !== today)
    return [...todayRows, ...rest]
  }, [trades, outcomeFilter, symbolSearch, sortCol, sortAsc, today])

  function toggleSort(col: string) {
    if (sortCol === col) setSortAsc(a => !a)
    else { setSortCol(col); setSortAsc(true) }
  }

  const winRateNum = stats ? parseFloat(stats.winRate) : 0
  const wrColor = winRateNum > 55 ? 'var(--win)' : winRateNum >= 45 ? 'var(--warn)' : 'var(--loss)'
  const OUTCOMES = ['WIN','LOSS','STOP']

  // ── MOBILE ──
  if (isMobile) {
    return (
      <div style={{ paddingBottom: 8 }}>
        {/* 2×3 stats grid */}
        {stats && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'rgba(255,255,255,0.05)', marginBottom: 12 }}>
            {[
              { label: 'Trades', val: `${stats.trades}`, color: 'var(--text-primary)' },
              { label: 'Win rate', val: `${stats.winRate}%`, color: wrColor },
              { label: 'Total P&L', val: fmtPnl(stats.totalPnl), color: pnlColor(stats.totalPnl) },
              { label: 'Avg / trade', val: fmtPnl(Math.round(stats.avgPnl)), color: pnlColor(stats.avgPnl) },
              { label: 'Best day', val: fmtPnl(Math.round(stats.bestDay)), color: 'var(--win)' },
              { label: 'Worst day', val: fmtPnl(Math.round(stats.worstDay)), color: 'var(--loss)' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ background: 'var(--bg-surface)', padding: '12px 14px' }}>
                <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2, fontFamily: 'var(--font-mono)' }}>{label}</div>
                <div className="mono" style={{ fontSize: 16, fontWeight: 500, color }}>{val}</div>
              </div>
            ))}
          </div>
        )}

        {/* Equity curve — compact height */}
        {equityCurve.length > 0 && (
          <div style={{ padding: '0 12px 12px' }}>
            <div style={{ background: 'var(--bg-surface)', borderRadius: 10, padding: '14px 12px 8px', border: 'var(--border)' }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 10 }}>Equity curve</div>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={equityCurve} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="eqm" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f24100" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#f24100" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#969696', fontFamily: 'var(--font-mono)' }}
                    tickFormatter={d => d.slice(5)} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis hide />
                  <ReferenceLine y={100000} stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" strokeWidth={1} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 11 }}
                    formatter={(val) => [fmtLakh(Number(val)), 'Capital']}
                  />
                  <Area type="monotone" dataKey="capital" stroke="#f24100" strokeWidth={2} fill="url(#eqm)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Filter pills */}
        <div style={{ padding: '0 12px 10px', display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          {['All', ...OUTCOMES].map(o => {
            const active = o === 'All' ? outcomeFilter.length === 0 : outcomeFilter.includes(o)
            return (
              <button key={o} onClick={() => {
                if (o === 'All') setOutcomeFilter([])
                else setOutcomeFilter(prev => prev.includes(o) ? prev.filter(x => x !== o) : [...prev, o])
              }} style={{
                padding: '3px 10px', borderRadius: 9999, fontSize: 11,
                border: `1px solid ${active ? 'var(--accent)' : 'rgba(255,255,255,0.1)'}`,
                background: active ? 'rgba(242,65,0,0.1)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: 'pointer', fontFamily: 'var(--font-mono)',
              }}>{o}</button>
            )
          })}
        </div>

        {/* Trade list (replaces table on mobile) */}
        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {trades.length === 0 && (
            <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>No paper trades recorded yet</div>
          )}
          {filtered.map((t, i) => {
            const pnl = parseFloat(t.pnl_inr)
            const pnlPct = parseFloat(t.pnl_pct)
            const outcomeCls = t.outcome === 'WIN' ? 'pill-win' : t.outcome === 'LOSS' ? 'pill-loss' : 'pill-stop'
            return (
              <div key={i} style={{
                background: 'var(--bg-surface)', borderRadius: 10,
                border: t.date === today ? '1px solid rgba(242,65,0,0.3)' : 'var(--border)',
                borderLeft: t.date === today ? '3px solid var(--accent)' : undefined,
                padding: '12px 14px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span className="mono" style={{ fontSize: 15, fontWeight: 500 }}>{t.symbol}</span>
                  <span className={`pill ${outcomeCls}`} style={{ fontSize: 10 }}>{t.outcome}</span>
                  <span className="mono" style={{ fontSize: 15, fontWeight: 500, color: pnlColor(pnl) }}>{fmtPnl(pnl)}</span>
                </div>
                <div style={{ display: 'flex', gap: 12, marginBottom: 6 }}>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {new Date(t.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}
                  </span>
                  <span className="mono" style={{ fontSize: 11 }}>₹{parseFloat(t.entry).toLocaleString('en-IN')}</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--loss)' }}>SL ₹{parseFloat(t.stop).toLocaleString('en-IN')}</span>
                  <span className="mono" style={{ fontSize: 11, color: pnlColor(pnlPct) }}>{pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // ── DESKTOP ──
  return (
    <div style={{ padding: '32px', paddingBottom: 48 }}>
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 32 }}>
          {[
            { label: 'Trades', val: `${stats.trades}`, color: 'var(--text-primary)' },
            { label: 'Win rate', val: `${stats.winRate}%`, color: wrColor },
            { label: 'Total P&L', val: fmtPnl(stats.totalPnl), color: pnlColor(stats.totalPnl) },
            { label: 'Avg P&L / trade', val: fmtPnl(Math.round(stats.avgPnl)), color: pnlColor(stats.avgPnl) },
            { label: 'Best day', val: fmtPnl(Math.round(stats.bestDay)), color: 'var(--win)' },
            { label: 'Worst day', val: fmtPnl(Math.round(stats.worstDay)), color: 'var(--loss)' },
          ].map(({ label, val, color }) => (
            <div key={label} className="metric-card">
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{label}</div>
              <div className="mono" style={{ fontSize: 20, fontWeight: 500, color }}>{val}</div>
            </div>
          ))}
        </div>
      )}
      {equityCurve.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 16 }}>Equity curve</h2>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={equityCurve} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f24100" stopOpacity={0.08} />
                  <stop offset="95%" stopColor="#f24100" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#969696', fontFamily: 'var(--font-mono)' }}
                tickFormatter={d => d.slice(5)} axisLine={false} tickLine={false} />
              <YAxis orientation="right" tick={{ fontSize: 11, fill: '#969696', fontFamily: 'var(--font-mono)' }}
                tickFormatter={v => fmtLakh(v)} axisLine={false} tickLine={false} width={72} />
              <ReferenceLine y={100000} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" strokeWidth={1} />
              <Tooltip
                contentStyle={{ background: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}
                labelStyle={{ color: '#969696' }}
                formatter={(val) => [fmtLakh(Number(val)), 'Capital']}
              />
              <Area type="monotone" dataKey="capital" stroke="#f24100" strokeWidth={2} fill="url(#eq)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="card" style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 20 }}>P&L calendar</h2>
        <CalendarHeatmap trades={trades} />
      </div>
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: 'var(--border)', display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)', marginRight: 4 }}>Outcome:</span>
          {['All', ...OUTCOMES].map(o => {
            const active = o === 'All' ? outcomeFilter.length === 0 : outcomeFilter.includes(o)
            return (
              <button key={o} onClick={() => {
                if (o === 'All') setOutcomeFilter([])
                else setOutcomeFilter(prev => prev.includes(o) ? prev.filter(x => x !== o) : [...prev, o])
              }} style={{
                padding: '3px 12px', borderRadius: 9999, fontSize: 11, border: `1px solid ${active ? 'var(--accent)' : 'rgba(255,255,255,0.1)'}`,
                background: active ? 'rgba(242,65,0,0.1)' : 'transparent', color: active ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: 'pointer', fontFamily: 'var(--font-mono)',
              }}>{o}</button>
            )
          })}
          <input value={symbolSearch} onChange={e => setSymbolSearch(e.target.value)} placeholder="Search symbol…"
            style={{ marginLeft: 'auto', padding: '4px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)', background: 'var(--bg-page)', color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', outline: 'none' }} />
        </div>
        {trades.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 14 }}>No paper trades recorded yet</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="trade-table">
              <thead>
                <tr>
                  {[['date','Date'],['symbol','Symbol'],['direction','Dir'],['entry','Entry ₹'],['stop','Stop ₹'],['high','High ₹'],['low','Low ₹'],['close','Close ₹'],['outcome','Outcome'],['pnl_inr','P&L ₹'],['pnl_pct','P&L %'],['regime','Regime'],['exposure','Exp'],['capital_after','Capital ₹']].map(([col, label]) => (
                    <th key={col} onClick={() => toggleSort(col)} style={{ cursor: 'pointer', userSelect: 'none' }}>
                      {label}{sortCol === col ? (sortAsc ? ' ↑' : ' ↓') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((t, i) => {
                  const pnl = parseFloat(t.pnl_inr)
                  const pnlPct = parseFloat(t.pnl_pct)
                  const isToday = t.date === today
                  return (
                    <tr key={i}>
                      <td className="mono" style={{ fontSize: 12, borderLeft: isToday ? '2px solid var(--accent)' : undefined }}>
                        {new Date(t.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}
                      </td>
                      <td className="mono" style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{t.symbol}</td>
                      <td><span className="pill pill-win" style={{ fontSize: 10 }}>LONG ↑</span></td>
                      <td className="mono" style={{ fontSize: 13 }}>₹{parseFloat(t.entry).toLocaleString('en-IN')}</td>
                      <td className="mono" style={{ fontSize: 13, color: 'var(--loss)' }}>₹{parseFloat(t.stop).toLocaleString('en-IN')}</td>
                      <td className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>₹{parseFloat(t.high).toLocaleString('en-IN')}</td>
                      <td className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>₹{parseFloat(t.low).toLocaleString('en-IN')}</td>
                      <td className="mono" style={{ fontSize: 12 }}>₹{parseFloat(t.close).toLocaleString('en-IN')}</td>
                      <td>{outcomePill(t.outcome)}</td>
                      <td className="mono" style={{ fontSize: 13, color: pnlColor(pnl) }}>{fmtPnl(pnl)}</td>
                      <td className="mono" style={{ fontSize: 12, color: pnlColor(pnlPct) }}>{pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%</td>
                      <td>{regimePill(t.regime)}</td>
                      <td className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{t.exposure}%</td>
                      <td className="mono" style={{ fontSize: 12 }}>₹{parseFloat(t.capital_after).toLocaleString('en-IN')}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
