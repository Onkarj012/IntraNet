'use client'
import { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import { useIsMobile } from '@/hooks/useIsMobile'

interface Fold { fold: number; period_start: string; period_end: string; dir_auc: number; meta_auc: number; sharpe: number; pnl_inr: number; win_rate: number; result: string }
interface WFData { folds: Fold[]; aggregate: { dir_auc: number; meta_auc: number; sharpe: number; pnl_inr: number; win_rate: number; positive_folds: number; total_folds: number; annual_return_pct: number; annual_return_slippage_pct: number; sharpe_after_slippage: number }; futures: { pnl_inr: number; sharpe: number; win_rate: number; profit_factor: number; trades: number; status: string } }
interface RegimeData { regimes: { regime: string; trades: number; win_rate: number; avg_pnl: number; sharpe: number; exposure: number }[] }
interface Trade { date: string; pnl_inr: string; capital_after: string }

function fmtPnl(v: number) {
  const abs = Math.abs(v)
  const s = abs >= 100000 ? `₹${(abs / 100000).toFixed(2)}L` : `₹${abs.toLocaleString('en-IN')}`
  return v >= 0 ? `+${s}` : `−${s}`
}
function pnlColor(v: number) { return v >= 0 ? 'var(--win)' : 'var(--loss)' }
function aucColor(v: number) { return v > 0.55 ? 'var(--win)' : v < 0.52 ? 'var(--loss)' : 'var(--warn)' }

function regimePillStyle(regime: string): { bg: string; color: string; border: string } {
  if (regime.includes('trending_bull')) return { bg: 'rgba(34,197,94,0.15)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.3)' }
  if (regime.includes('trending_bear')) return { bg: 'rgba(239,68,68,0.15)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }
  if (regime.includes('choppy')) return { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)' }
  if (regime.includes('crisis')) return { bg: 'rgba(239,68,68,0.15)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }
  return { bg: 'rgba(150,150,150,0.15)', color: '#969696', border: '1px solid rgba(150,150,150,0.3)' }
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
function monthReturnColor(pct: number) {
  if (pct > 3)  return '#166534'
  if (pct > 1)  return '#15803d'
  if (pct > 0)  return '#14532d'
  if (pct > -1) return '#7f1d1d'
  if (pct > -3) return '#991b1b'
  return '#450a0a'
}

export default function PerformancePage() {
  const [wf, setWf] = useState<WFData | null>(null)
  const [regime, setRegime] = useState<RegimeData | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const isMobile = useIsMobile()

  useEffect(() => {
    fetch('/api/walk-forward').then(r => r.json()).then(setWf)
    fetch('/api/regime').then(r => r.json()).then(setRegime)
    fetch('/api/ledger').then(r => r.text()).then(csv => {
      setTrades(Papa.parse<Trade>(csv, { header: true, skipEmptyLines: true }).data)
    })
  }, [])

  // Monthly returns matrix
  const monthlyMatrix = useMemo(() => {
    if (!trades.length) return {}
    const m: Record<string, Record<number, { pnl: number; startCap: number }>> = {}
    for (const t of trades) {
      const d = new Date(t.date)
      const y = d.getFullYear(), mo = d.getMonth()
      if (!m[y]) m[y] = {}
      if (!m[y][mo]) m[y][mo] = { pnl: 0, startCap: 100000 }
      m[y][mo].pnl += parseFloat(t.pnl_inr)
    }
    return m
  }, [trades])

  const years = Object.keys(monthlyMatrix).sort()

  if (!wf || !regime) return <div style={{ padding: 32, color: 'var(--text-secondary)' }}>Loading…</div>
  const { aggregate: agg, folds, futures } = wf

  // ── MOBILE ──
  if (isMobile) {
    return (
      <div style={{ paddingBottom: 8 }}>
        {/* Hero metrics — stacked */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'rgba(255,255,255,0.05)', marginBottom: 12 }}>
          <div style={{ background: 'var(--bg-surface)', padding: '14px', gridColumn: '1/-1', borderLeft: '3px solid var(--accent)' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>Walk-forward Sharpe</div>
            <div className="mono" style={{ fontSize: 24, fontWeight: 500 }}>{agg.sharpe.toFixed(2)}</div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>{agg.sharpe_after_slippage.toFixed(2)} after slippage</div>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: '12px 14px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>Positive folds</div>
            <div className="mono" style={{ fontSize: 20, fontWeight: 500 }}>{agg.positive_folds}/{agg.total_folds}</div>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: '12px 14px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>Annual return</div>
            <div className="mono" style={{ fontSize: 20, fontWeight: 500, color: 'var(--win)' }}>+{agg.annual_return_slippage_pct}%</div>
          </div>
        </div>

        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Fold table — horizontal scroll */}
          <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', overflow: 'hidden' }}>
            <div style={{ padding: '12px 14px', borderBottom: 'var(--border)', fontSize: 12, fontWeight: 500 }}>
              Out-of-sample · {folds.length} folds
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="trade-table" style={{ fontSize: 11 }}>
                <thead>
                  <tr>{['#','Period','Dir AUC','Sharpe','P&L','WR',''].map(h => <th key={h} style={{ fontSize: 10 }}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {folds.map(f => (
                    <tr key={f.fold} style={{ borderLeft: `2px solid ${f.result === 'POSITIVE' ? 'var(--win)' : 'var(--loss)'}` }}>
                      <td className="mono">{f.fold}</td>
                      <td className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                        {new Date(f.period_start).toLocaleDateString('en-IN',{month:'short',year:'2-digit'})}
                      </td>
                      <td className="mono" style={{ color: aucColor(f.dir_auc) }}>{f.dir_auc.toFixed(3)}</td>
                      <td className="mono" style={{ color: pnlColor(f.sharpe) }}>{f.sharpe.toFixed(2)}</td>
                      <td className="mono" style={{ color: pnlColor(f.pnl_inr), whiteSpace: 'nowrap' }}>{fmtPnl(f.pnl_inr)}</td>
                      <td className="mono" style={{ color: f.win_rate > 55 ? 'var(--win)' : f.win_rate < 45 ? 'var(--loss)' : 'var(--warn)' }}>{f.win_rate.toFixed(0)}%</td>
                      <td><span className={`pill ${f.result === 'POSITIVE' ? 'pill-win' : 'pill-loss'}`} style={{ fontSize: 9 }}>{f.result === 'POSITIVE' ? '✓' : '✗'}</span></td>
                    </tr>
                  ))}
                  <tr style={{ background: '#1a1a1a' }}>
                    <td className="mono" style={{ fontWeight: 500 }}>Agg</td>
                    <td className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>2Y</td>
                    <td className="mono" style={{ color: aucColor(agg.dir_auc) }}>{agg.dir_auc.toFixed(3)}</td>
                    <td className="mono" style={{ color: 'var(--win)' }}>{agg.sharpe.toFixed(2)}</td>
                    <td className="mono" style={{ color: 'var(--win)', whiteSpace: 'nowrap' }}>{fmtPnl(agg.pnl_inr)}</td>
                    <td className="mono" style={{ color: 'var(--win)' }}>{agg.win_rate.toFixed(0)}%</td>
                    <td className="mono" style={{ fontSize: 10 }}>{agg.positive_folds}/{agg.total_folds}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Regime cards — single column */}
          <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', padding: '14px' }}>
            <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 12 }}>By regime</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {regime.regimes.filter(r => r.trades > 0).map(r => {
                const ps = regimePillStyle(r.regime)
                return (
                  <div key={r.regime} style={{ borderBottom: 'var(--border)', paddingBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <span className="pill" style={{ ...ps, fontSize: 10 }}>{r.regime.replace(/_/g, ' ')}</span>
                      <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{r.trades} trades · {r.win_rate.toFixed(0)}% win</span>
                    </div>
                    <div style={{ display: 'flex', gap: 16 }}>
                      <span className="mono" style={{ fontSize: 12, color: pnlColor(r.avg_pnl) }}>avg {r.avg_pnl >= 0 ? '+' : ''}₹{r.avg_pnl}</span>
                      <span className="mono" style={{ fontSize: 12, color: r.sharpe > 1 ? 'var(--win)' : 'var(--warn)' }}>S={r.sharpe.toFixed(2)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Monthly returns — horizontal scroll */}
          {years.length > 0 && (
            <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', overflow: 'hidden' }}>
              <div style={{ padding: '12px 14px', borderBottom: 'var(--border)', fontSize: 12, fontWeight: 500 }}>Monthly returns</div>
              <div style={{ overflowX: 'auto', padding: '12px' }}>
                <table style={{ borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '4px 8px', fontSize: 10, color: 'var(--text-secondary)', fontWeight: 400, textAlign: 'left' }}></th>
                      {MONTHS.map(m => <th key={m} style={{ padding: '4px 4px', fontSize: 10, color: 'var(--text-secondary)', fontWeight: 400, fontFamily: 'var(--font-mono)', textAlign: 'center', minWidth: 36 }}>{m}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {years.map(y => {
                      const yearData = monthlyMatrix[y]
                      return (
                        <tr key={y}>
                          <td style={{ padding: '2px 8px', fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>{y}</td>
                          {MONTHS.map((_, mi) => {
                            const d = yearData[mi]
                            const pct = d ? d.pnl / 100000 * 100 : null
                            return (
                              <td key={mi} style={{ padding: '2px 2px', textAlign: 'center' }}>
                                <div style={{ padding: '4px 0', borderRadius: 3, background: pct !== null ? monthReturnColor(pct) : 'rgba(255,255,255,0.03)', fontSize: 9, fontFamily: 'var(--font-mono)', color: pct !== null ? 'rgba(255,255,255,0.8)' : 'var(--text-muted)', minWidth: 32, textAlign: 'center' }}>
                                  {pct !== null ? `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}` : '—'}
                                </div>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* System comparison — stacked */}
          {[
            { label: 'Equity intraday', status: 'ACTIVE', statusCls: 'pill-win', border: 'rgba(34,197,94,0.3)', pnl: agg.pnl_inr, sharpe: agg.sharpe, winRate: agg.win_rate, extra: `${agg.positive_folds}/${agg.total_folds} folds +ve` },
            { label: 'NIFTY futures', status: 'SOFT HALT', statusCls: 'pill-stop', border: 'rgba(239,68,68,0.3)', pnl: futures.pnl_inr, sharpe: futures.sharpe, winRate: futures.win_rate, extra: `PF ${futures.profit_factor.toFixed(2)}` },
          ].map(sys => (
            <div key={sys.label} style={{ background: 'var(--bg-surface)', borderRadius: 10, border: `1px dashed ${sys.border}`, padding: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>{sys.label}</span>
                <span className={`pill ${sys.statusCls}`} style={{ fontSize: 10 }}>{sys.status}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                {[['P&L', fmtPnl(sys.pnl), pnlColor(sys.pnl)], ['Sharpe', sys.sharpe.toFixed(2), 'var(--text-primary)'], ['Win rate', `${sys.winRate.toFixed(1)}%`, sys.winRate > 55 ? 'var(--win)' : 'var(--warn)']].map(([l, v, c]) => (
                  <div key={l}>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>{l}</div>
                    <div className="mono" style={{ fontSize: 13, color: c as string }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{sys.extra}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // ── DESKTOP ──
  return (
    <div style={{ padding: '32px', paddingBottom: 48 }}>
      {/* Health header */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 32 }}>
        {[
          { label: 'Positive folds', val: `${agg.positive_folds} / ${agg.total_folds}`, sub: 'Out-of-sample' },
          { label: 'Walk-forward Sharpe', val: agg.sharpe.toFixed(2), sub: `${agg.sharpe_after_slippage.toFixed(2)} after slippage` },
          { label: 'Annual return (w/ slippage)', val: `+${agg.annual_return_slippage_pct}%`, sub: `${agg.annual_return_pct}% gross` },
        ].map(({ label, val, sub }) => (
          <div key={label} className="card" style={{ borderLeft: '2px solid var(--accent)', textAlign: 'center' }}>
            <div className="mono" style={{ fontSize: 28, fontWeight: 500, color: 'var(--text-primary)', marginBottom: 6 }}>{val}</div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{sub}</div>
          </div>
        ))}
      </div>

      {/* Walk-forward fold table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: 24 }}>
        <div style={{ padding: '18px 20px', borderBottom: 'var(--border)' }}>
          <h2 style={{ fontSize: 16, fontWeight: 500 }}>Out-of-sample validation · {folds.length} folds</h2>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="trade-table">
            <thead>
              <tr>
                {['Fold','Period','Dir AUC','Meta AUC','Sharpe','P&L ₹','Win rate','Result'].map(h => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {folds.map(f => (
                <tr key={f.fold} style={{ borderLeft: `2px solid ${f.result === 'POSITIVE' ? 'var(--win)' : 'var(--loss)'}` }}>
                  <td className="mono">{f.fold}</td>
                  <td className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {new Date(f.period_start).toLocaleDateString('en-IN',{month:'short',year:'numeric'})} – {new Date(f.period_end).toLocaleDateString('en-IN',{month:'short',year:'numeric'})}
                  </td>
                  <td className="mono" style={{ color: aucColor(f.dir_auc) }}>{f.dir_auc.toFixed(3)}</td>
                  <td className="mono" style={{ color: aucColor(f.meta_auc) }}>{f.meta_auc.toFixed(3)}</td>
                  <td className="mono" style={{ color: pnlColor(f.sharpe) }}>{f.sharpe.toFixed(2)}</td>
                  <td className="mono" style={{ color: pnlColor(f.pnl_inr) }}>{fmtPnl(f.pnl_inr)}</td>
                  <td className="mono" style={{ color: f.win_rate > 55 ? 'var(--win)' : f.win_rate < 45 ? 'var(--loss)' : 'var(--warn)' }}>{f.win_rate.toFixed(1)}%</td>
                  <td><span className={`pill ${f.result === 'POSITIVE' ? 'pill-win' : 'pill-loss'}`}>{f.result === 'POSITIVE' ? '✓' : '✗'}</span></td>
                </tr>
              ))}
              <tr style={{ background: '#1a1a1a', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <td className="mono" style={{ fontWeight: 500 }}>Agg</td>
                <td className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>2Y total</td>
                <td className="mono" style={{ color: aucColor(agg.dir_auc) }}>{agg.dir_auc.toFixed(3)}</td>
                <td className="mono" style={{ color: aucColor(agg.meta_auc) }}>{agg.meta_auc.toFixed(3)}</td>
                <td className="mono" style={{ color: 'var(--win)' }}>{agg.sharpe.toFixed(2)}</td>
                <td className="mono" style={{ color: 'var(--win)' }}>{fmtPnl(agg.pnl_inr)}</td>
                <td className="mono" style={{ color: 'var(--win)' }}>{agg.win_rate.toFixed(1)}%</td>
                <td><span className="mono" style={{ fontSize: 13 }}>{agg.positive_folds}/{agg.total_folds} ✓</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Regime breakdown */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 20 }}>Performance by regime</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {regime.regimes.map(r => {
            const ps = regimePillStyle(r.regime)
            return (
              <div key={r.regime} style={{ background: 'var(--bg-page)', borderRadius: 8, padding: 16, border: 'var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 14 }}>
                  <span className="pill" style={{ ...ps, fontSize: 11 }}>{r.regime.replace(/_/g, ' ')}</span>
                  <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{r.trades} trades</span>
                </div>
                {r.trades === 0 ? (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>No data</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', width: 72 }}>Win rate</span>
                      <div style={{ flex: 1, height: 4, borderRadius: 2, background: '#1a1a1a' }}>
                        <div style={{ height: '100%', borderRadius: 2, background: r.win_rate > 55 ? 'var(--win)' : r.win_rate < 45 ? 'var(--loss)' : 'var(--warn)', width: `${r.win_rate}%` }} />
                      </div>
                      <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', width: 36 }}>{r.win_rate.toFixed(0)}%</span>
                    </div>
                    {[
                      { label: 'Avg P&L', val: `${r.avg_pnl >= 0 ? '+' : ''}₹${r.avg_pnl}`, color: pnlColor(r.avg_pnl) },
                      { label: 'Sharpe', val: r.sharpe.toFixed(2), color: r.sharpe > 1 ? 'var(--win)' : 'var(--warn)' },
                      { label: 'Exposure', val: `${r.exposure}%`, color: 'var(--text-secondary)' },
                    ].map(({ label, val, color }) => (
                      <div key={label} style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{label}</span>
                        <span className="mono" style={{ fontSize: 13, color }}>{val}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Monthly returns matrix */}
      {years.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 16 }}>Monthly returns</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ padding: '6px 12px', fontSize: 11, color: 'var(--text-secondary)', textAlign: 'left', fontWeight: 400 }}></th>
                  {MONTHS.map(m => <th key={m} style={{ padding: '6px 8px', fontSize: 11, color: 'var(--text-secondary)', fontWeight: 400, fontFamily: 'var(--font-mono)', textAlign: 'center' }}>{m}</th>)}
                  <th style={{ padding: '6px 12px', fontSize: 11, color: 'var(--text-secondary)', fontWeight: 400, fontFamily: 'var(--font-mono)', textAlign: 'center', borderLeft: 'var(--border-solid)' }}>Annual</th>
                </tr>
              </thead>
              <tbody>
                {years.map(y => {
                  const yearData = monthlyMatrix[y]
                  const annual = Object.values(yearData).reduce((s, d) => s + d.pnl, 0) / 100000 * 100
                  return (
                    <tr key={y}>
                      <td style={{ padding: '4px 12px', fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{y}</td>
                      {MONTHS.map((_, mi) => {
                        const d = yearData[mi]
                        const pct = d ? d.pnl / 100000 * 100 : null
                        return (
                          <td key={mi} style={{ padding: '2px 2px', textAlign: 'center' }}>
                            <div title={d ? `₹${d.pnl.toLocaleString('en-IN')}` : ''}
                              style={{ padding: '6px 0', borderRadius: 4, background: pct !== null ? monthReturnColor(pct) : 'rgba(255,255,255,0.03)', fontSize: 11, fontFamily: 'var(--font-mono)', color: pct !== null ? 'rgba(255,255,255,0.8)' : 'var(--text-muted)', cursor: pct !== null ? 'default' : 'default', minWidth: 44, textAlign: 'center' }}>
                              {pct !== null ? `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%` : '—'}
                            </div>
                          </td>
                        )
                      })}
                      <td style={{ padding: '2px 4px', textAlign: 'center', borderLeft: 'var(--border-solid)' }}>
                        <div style={{ padding: '6px 8px', borderRadius: 4, background: monthReturnColor(annual), fontSize: 11, fontFamily: 'var(--font-mono)', color: 'rgba(255,255,255,0.8)' }}>
                          {annual >= 0 ? '+' : ''}{annual.toFixed(1)}%
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Equity vs Futures comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Equity */}
        <div style={{ background: 'var(--bg-surface)', borderRadius: 12, padding: 24, border: '1px dashed rgba(34,197,94,0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
            <span style={{ fontSize: 16, fontWeight: 500 }}>Equity intraday</span>
            <span className="pill pill-win">ACTIVE</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { label: 'P&L', val: fmtPnl(agg.pnl_inr), color: pnlColor(agg.pnl_inr) },
              { label: 'Sharpe (WF)', val: agg.sharpe.toFixed(2), color: 'var(--text-primary)' },
              { label: 'Win rate', val: `${agg.win_rate.toFixed(1)}%`, color: 'var(--win)' },
              { label: 'Trades', val: `${trades.length} live`, color: 'var(--text-secondary)' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: 'var(--border)', paddingBottom: 8 }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
                <span className="mono" style={{ fontSize: 13, color }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-page)', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>Walk-forward</div>
              <div className="mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>{agg.positive_folds}/{agg.total_folds} folds +ve · {fmtPnl(agg.pnl_inr)} / 2yr</div>
            </div>
          </div>
        </div>
        {/* Futures */}
        <div style={{ background: 'var(--bg-surface)', borderRadius: 12, padding: 24, border: '1px dashed rgba(239,68,68,0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
            <span style={{ fontSize: 16, fontWeight: 500 }}>NIFTY futures</span>
            <span className="pill pill-stop">SOFT HALT</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { label: 'P&L', val: fmtPnl(futures.pnl_inr), color: pnlColor(futures.pnl_inr) },
              { label: 'Sharpe (WF)', val: `${futures.sharpe.toFixed(2)} (soft halt)`, color: 'var(--warn)' },
              { label: 'Win rate', val: `${futures.win_rate.toFixed(1)}%`, color: 'var(--warn)' },
              { label: 'Profit factor', val: futures.profit_factor.toFixed(2), color: 'var(--text-primary)' },
              { label: 'Trades', val: `${futures.trades} live`, color: 'var(--text-secondary)' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: 'var(--border)', paddingBottom: 8 }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
                <span className="mono" style={{ fontSize: 13, color }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-page)', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>Walk-forward</div>
              <div className="mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>{fmtPnl(futures.pnl_inr)}, S={futures.sharpe.toFixed(2)} · {futures.win_rate.toFixed(1)}% win, PF {futures.profit_factor.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
