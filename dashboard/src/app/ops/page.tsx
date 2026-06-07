'use client'
import { useEffect, useState } from 'react'
import { useIsMobile } from '@/hooks/useIsMobile'

interface Step { label: string; return_code: number; duration: number; stdout: string; stderr: string }
interface StatusData { last_run_mode: string; timestamp: string; total_duration: number; steps_passed: number; steps_total: number; steps: Step[] }
interface TrainMeta { trained_at: string; dir_auc: number; meta_auc: number; model_version: string; retrain_threshold_days: number }
interface DriftData { current_sharpe: number; sharpe_threshold: number; win_rate: number; win_rate_threshold: number; drift_level: string; equity_halt_active: boolean; futures_halt_active: boolean; halt_file_exists: boolean }

function fmtDuration(s: number) {
  const m = Math.floor(s / 60), sec = s % 60
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}
function fmtIST(iso: string) {
  return new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'medium', hour12: false }).format(new Date(iso))
}
function daysSince(iso: string) {
  return Math.floor((Date.now() - new Date(iso).getTime()) / 86400000)
}
function aucColor(v: number) { return v > 0.55 ? 'var(--win)' : v < 0.52 ? 'var(--loss)' : 'var(--warn)' }

function DriftGauge({ label, current, threshold, isPercent }: { label: string; current: number; threshold: number; isPercent?: boolean }) {
  const max = isPercent ? 100 : 3
  const pct = Math.min(current / max, 1)
  const thPct = Math.min(threshold / max, 1)
  const ok = current >= threshold
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 13, color: ok ? 'var(--win)' : 'var(--loss)' }}>
            {isPercent ? `${current.toFixed(1)}%` : current.toFixed(2)} {ok ? '▲' : '▼'} threshold ({isPercent ? `${threshold}%` : threshold})
          </span>
          <span className={`pill ${ok ? 'pill-win' : 'pill-loss'}`}>{ok ? 'OK' : 'WARN'}</span>
        </div>
      </div>
      <div style={{ width: 200, height: 6, borderRadius: 3, background: '#1a1a1a', position: 'relative' }}>
        <div style={{ height: '100%', borderRadius: 3, background: ok ? 'var(--win)' : 'var(--loss)', width: `${pct * 100}%` }} />
        <div style={{ position: 'absolute', top: -2, left: `${thPct * 100}%`, width: 2, height: 10, background: 'rgba(255,255,255,0.6)', borderStyle: 'dashed', borderWidth: '0 1px' }} />
      </div>
    </div>
  )
}

export default function OpsPage() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [trainMeta, setTrainMeta] = useState<TrainMeta | null>(null)
  const [drift, setDrift] = useState<DriftData | null>(null)
  const [confirm, setConfirm] = useState<null | { system: 'equity' | 'futures'; action: 'halt' | 'clear' }>(null)
  const isMobile = useIsMobile()

  useEffect(() => {
    fetch('/api/status').then(r => r.json()).then(setStatus)
    fetch('/api/train-meta').then(r => r.json()).then(setTrainMeta)
    fetch('/api/drift').then(r => r.json()).then(setDrift)
  }, [])

  async function toggleHalt(system: 'equity' | 'futures', action: 'halt' | 'clear') {
    if (!drift) return
    const key = system === 'equity' ? 'equity_halt_active' : 'futures_halt_active'
    const updated = { ...drift, [key]: action === 'halt' }
    const res = await fetch('/api/drift', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [key]: action === 'halt' }) })
    setDrift(await res.json())
    setConfirm(null)
  }

  if (!status || !trainMeta || !drift) return <div style={{ padding: 32, color: 'var(--text-secondary)' }}>Loading…</div>

  const allPassed = status.steps_passed === status.steps_total
  const days = daysSince(trainMeta.trained_at)
  const freshnessLabel = days < 14 ? 'FRESH' : days < 21 ? 'AGING' : 'STALE'
  const freshnessClass = days < 14 ? 'pill-win' : days < 21 ? 'pill-stop' : 'pill-loss'
  const nextRetrain = new Date(new Date(trainMeta.trained_at).getTime() + trainMeta.retrain_threshold_days * 86400000)

  // ── MOBILE ──
  if (isMobile) {
    return (
      <div style={{ paddingBottom: 8 }}>
        {/* Pipeline status */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'rgba(255,255,255,0.05)', marginBottom: 12 }}>
          <div style={{ background: 'var(--bg-surface)', padding: '12px 14px', gridColumn: '1/-1', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>Pipeline</span>
            <span className={`pill ${allPassed ? 'pill-win' : 'pill-loss'}`} style={{ fontSize: 10 }}>{allPassed ? 'ALL SYSTEMS GO' : 'FAILED'}</span>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: '10px 14px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>Steps</div>
            <div className="mono" style={{ fontSize: 14 }}>{status.steps_passed}/{status.steps_total}</div>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: '10px 14px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>Duration</div>
            <div className="mono" style={{ fontSize: 14 }}>{fmtDuration(status.total_duration)}</div>
          </div>
        </div>

        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Terminal log — compact */}
          <div style={{ borderRadius: 10, border: 'var(--border)', overflow: 'hidden' }}>
            <div style={{ background: '#1a1a1a', padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 8, borderBottom: 'var(--border)' }}>
              <div style={{ display: 'flex', gap: 4 }}>
                {['#ef4444','#f59e0b','#22c55e'].map(c => <div key={c} style={{ width: 9, height: 9, borderRadius: '50%', background: c, opacity: 0.8 }} />)}
              </div>
              <span className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>intraday_pipeline</span>
            </div>
            <div style={{ background: '#050505', padding: '10px 0' }}>
              {status.steps.map((step, i) => {
                const ok = step.return_code === 0
                return (
                  <div key={i}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 12px', background: !ok ? 'rgba(239,68,68,0.06)' : 'transparent' }}>
                      <span style={{ color: ok ? 'var(--win)' : 'var(--loss)', fontSize: 9 }}>{ok ? '●' : '✗'}</span>
                      <span className="mono" style={{ fontSize: 10, color: 'var(--text-primary)', flex: 1 }}>{step.label}</span>
                      <span className="mono" style={{ fontSize: 9, color: ok ? 'var(--text-secondary)' : 'var(--loss)' }}>[{step.return_code}]</span>
                    </div>
                    {!ok && step.stderr && (
                      <div style={{ padding: '4px 12px 6px 28px', background: 'rgba(239,68,68,0.04)' }}>
                        <div className="mono" style={{ fontSize: 9, color: 'var(--loss)' }}>{step.stderr}</div>
                      </div>
                    )}
                  </div>
                )
              })}
              <div style={{ padding: '6px 12px', display: 'flex' }}>
                <span style={{ display: 'inline-block', width: 2, height: 12, background: 'var(--accent)' }} className="cursor-blink" />
              </div>
            </div>
          </div>

          {/* Model health */}
          <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 500 }}>Model health</span>
              <span className={`pill ${freshnessClass}`} style={{ fontSize: 10 }}>{freshnessLabel}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                ['Days since', `${days}d ago`],
                ['Dir AUC', trainMeta.dir_auc.toFixed(3)],
                ['Meta AUC', trainMeta.meta_auc.toFixed(3)],
                ['Next retrain', nextRetrain.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })],
              ].map(([label, val]) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 }}>{label}</div>
                  <div className="mono" style={{ fontSize: 13 }}>{val}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Drift gauges */}
          <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', padding: '14px' }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Drift monitor</div>
            <DriftGauge label="Sharpe (20d)" current={drift.current_sharpe} threshold={drift.sharpe_threshold} />
            <DriftGauge label="Win rate (20d)" current={drift.win_rate} threshold={drift.win_rate_threshold} isPercent />
            <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 8 }}>
              Status: <span style={{ color: drift.drift_level === 'OK' ? 'var(--win)' : 'var(--loss)' }}>{drift.drift_level}</span>
              {' · '}Soft halt: <span style={{ color: drift.futures_halt_active ? 'var(--loss)' : 'var(--text-secondary)' }}>{drift.futures_halt_active ? 'active' : 'off'}</span>
            </div>
          </div>

          {/* Halt controls */}
          <div style={{ background: 'var(--bg-surface)', borderRadius: 10, border: 'var(--border)', padding: '14px' }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Kill switches</div>
            {(['equity', 'futures'] as const).map(sys => {
              const halted = sys === 'equity' ? drift.equity_halt_active : drift.futures_halt_active
              return (
                <div key={sys} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, padding: '10px 12px', background: 'var(--bg-page)', borderRadius: 8 }}>
                  <div>
                    <div style={{ fontSize: 12, textTransform: 'capitalize', marginBottom: 4 }}>{sys} system</div>
                    <span className={`pill ${halted ? 'pill-loss' : 'pill-win'}`} style={{ fontSize: 10 }}>{halted ? '○ HALTED' : '● ARMED'}</span>
                  </div>
                  <button onClick={() => setConfirm({ system: sys, action: halted ? 'clear' : 'halt' })}
                    style={{ padding: '5px 14px', borderRadius: 6, fontSize: 11, cursor: 'pointer', border: `1px solid ${halted ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`, background: 'transparent', color: halted ? 'var(--win)' : 'var(--loss)', fontFamily: 'var(--font-mono)' }}>
                    {halted ? 'Clear' : 'Halt'}
                  </button>
                </div>
              )
            })}
          </div>
        </div>

        {/* Confirm dialog */}
        {confirm && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'flex-end', justifyContent: 'center', zIndex: 100, padding: '0 0 72px' }}>
            <div style={{ background: 'var(--bg-surface)', border: 'var(--border)', borderRadius: 12, padding: 24, width: '90%', maxWidth: 400 }}>
              <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 10 }}>Confirm {confirm.action === 'halt' ? 'halt' : 'clear halt'}</div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20 }}>
                {confirm.action === 'halt' ? `Block all ${confirm.system} entries?` : `Re-arm ${confirm.system} system?`}
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <button onClick={() => setConfirm(null)} style={{ flex: 1, padding: '10px', borderRadius: 8, border: 'var(--border-solid)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13 }}>Cancel</button>
                <button onClick={() => toggleHalt(confirm.system, confirm.action)}
                  style={{ flex: 1, padding: '10px', borderRadius: 8, border: `1px solid ${confirm.action === 'halt' ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)'}`, background: confirm.action === 'halt' ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)', color: confirm.action === 'halt' ? 'var(--loss)' : 'var(--win)', cursor: 'pointer', fontSize: 13 }}>
                  Confirm
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // ── DESKTOP ──
  return (
    <div style={{ padding: 32, paddingBottom: 48 }}>
      {/* Pipeline run status */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 500 }}>Last pipeline run</h2>
          <span className={`pill ${allPassed ? 'pill-win' : 'pill-loss'}`}>
            {allPassed ? 'ALL SYSTEMS GO' : 'PIPELINE FAILED'}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {[
            { label: 'Mode', val: status.last_run_mode },
            { label: 'Started', val: fmtIST(status.timestamp) },
            { label: 'Duration', val: fmtDuration(status.total_duration) },
            { label: 'Steps', val: `${status.steps_passed} / ${status.steps_total} passed` },
          ].map(({ label, val }) => (
            <div key={label}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
              <div className="mono" style={{ fontSize: 13, color: 'var(--text-primary)' }}>{val}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Cron log terminal */}
      <div style={{ marginBottom: 24, border: 'var(--border)', borderRadius: 12, overflow: 'hidden' }}>
        {/* Terminal header */}
        <div style={{ background: '#1a1a1a', padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10, borderBottom: 'var(--border)' }}>
          <div style={{ display: 'flex', gap: 6 }}>
            {['#ef4444','#f59e0b','#22c55e'].map(c => <div key={c} style={{ width: 12, height: 12, borderRadius: '50%', background: c, opacity: 0.8 }} />)}
          </div>
          <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', marginLeft: 8 }}>
            intraday_pipeline · {new Date(status.timestamp).toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' })}
          </span>
        </div>
        {/* Steps */}
        <div style={{ background: '#050505', padding: '16px 0' }}>
          {status.steps.map((step, i) => {
            const ok = step.return_code === 0
            return (
              <div key={i}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '4px 20px', background: !ok ? 'rgba(239,68,68,0.06)' : 'transparent' }}>
                  <span style={{ color: ok ? 'var(--win)' : 'var(--loss)', fontSize: 10 }}>{ok ? '●' : '✗'}</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', width: 16 }}>{String(i).padStart(2,'0')}</span>
                  <span className="mono" style={{ fontSize: 12, color: 'var(--text-primary)', width: 220 }}>{step.label}</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', width: 48 }}>{step.duration.toFixed(1)}s</span>
                  <span className="mono" style={{ fontSize: 11, color: ok ? 'var(--text-secondary)' : 'var(--loss)' }}>[{step.return_code}]</span>
                  {ok && <span className="mono" style={{ fontSize: 11, color: 'var(--win)' }}>✓</span>}
                  {!ok && <span className="mono" style={{ fontSize: 11, color: 'var(--loss)' }}>FAILED</span>}
                </div>
                {!ok && (step.stderr || step.stdout) && (
                  <div style={{ padding: '6px 20px 8px 60px', background: 'rgba(239,68,68,0.04)' }}>
                    {step.stderr && <div className="mono" style={{ fontSize: 11, color: 'var(--loss)' }}>└─ {step.stderr}</div>}
                    {step.stdout && step.stdout.split('\n').map((line, li) => (
                      <div key={li} className="mono" style={{ fontSize: 11, color: 'var(--warn)', marginLeft: 16 }}>{line}</div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
          <div style={{ padding: '8px 20px', display: 'flex', alignItems: 'center', gap: 2 }}>
            <span style={{ display: 'inline-block', width: 2, height: 14, background: 'var(--accent)' }} className="cursor-blink" />
          </div>
        </div>
      </div>

      {/* Model freshness */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 500 }}>Model health</h2>
          <span className={`pill ${freshnessClass}`}>{freshnessLabel}</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Trained at</div>
            <div className="mono" style={{ fontSize: 13 }}>{fmtIST(trainMeta.trained_at)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Days since</div>
            <div className="mono" style={{ fontSize: 13 }}>{days} days ago</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Dir AUC</div>
            <div className="mono" style={{ fontSize: 13, color: aucColor(trainMeta.dir_auc) }}>{trainMeta.dir_auc.toFixed(3)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Meta AUC</div>
            <div className="mono" style={{ fontSize: 13, color: aucColor(trainMeta.meta_auc) }}>{trainMeta.meta_auc.toFixed(3)}</div>
          </div>
        </div>
        <div style={{ padding: '10px 14px', background: 'var(--bg-page)', borderRadius: 8, fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          Retrain threshold: {trainMeta.retrain_threshold_days} days · Next due: {nextRetrain.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
        </div>
      </div>

      {/* Drift monitor & halt controls */}
      <div className="card">
        <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 20 }}>Drift monitor & halt controls</h2>

        {/* Gauges */}
        <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: 'var(--border)' }}>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>Current performance vs thresholds</div>
          <DriftGauge label="Sharpe (rolling 20d)" current={drift.current_sharpe} threshold={drift.sharpe_threshold} />
          <DriftGauge label="Win rate (rolling 20d)" current={drift.win_rate} threshold={drift.win_rate_threshold} isPercent />
          <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Drift status: <span style={{ color: drift.drift_level === 'OK' ? 'var(--win)' : 'var(--loss)' }}>{drift.drift_level}</span>
            </span>
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Soft halt: <span style={{ color: drift.futures_halt_active ? 'var(--loss)' : 'var(--text-secondary)' }}>{drift.futures_halt_active ? 'active' : 'inactive'}</span>
            </span>
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Hard halt: <span style={{ color: drift.halt_file_exists ? 'var(--loss)' : 'var(--text-secondary)' }}>{drift.halt_file_exists ? 'active' : 'inactive'}</span>
            </span>
          </div>
        </div>

        {/* Kill switches */}
        <div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>Kill switches</div>
          {([['equity', drift.equity_halt_active], ['futures', drift.futures_halt_active]] as const).map(([sys, halted]) => (
            <div key={sys} style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14, padding: '12px 16px', background: 'var(--bg-page)', borderRadius: 8 }}>
              <span style={{ fontSize: 13, color: 'var(--text-primary)', textTransform: 'capitalize', width: 120 }}>{sys} system</span>
              <span className={`pill ${halted ? 'pill-loss' : 'pill-win'}`}>
                {halted ? '○ HALTED — entries blocked' : '● ARMED — entries allowed'}
              </span>
              <button onClick={() => setConfirm({ system: sys, action: halted ? 'clear' : 'halt' })}
                style={{
                  marginLeft: 'auto', padding: '5px 16px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
                  border: `1px solid ${halted ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`,
                  background: 'transparent',
                  color: halted ? 'var(--win)' : 'var(--loss)',
                  fontFamily: 'var(--font-mono)',
                }}>
                {halted ? 'Clear halt' : 'Trigger halt'}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Confirmation dialog */}
      {confirm && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ background: 'var(--bg-surface)', border: 'var(--border)', borderRadius: 12, padding: 32, maxWidth: 400, width: '90%' }}>
            <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 12 }}>Confirm {confirm.action === 'halt' ? 'halt' : 'clear halt'}</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 24 }}>
              {confirm.action === 'halt'
                ? `This will block all ${confirm.system} entries. Paper mode only — no real trade routing.`
                : `This will re-arm the ${confirm.system} system and allow entries.`
              }
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button onClick={() => setConfirm(null)} style={{ padding: '6px 20px', borderRadius: 6, border: 'var(--border-solid)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13 }}>Cancel</button>
              <button onClick={() => toggleHalt(confirm.system, confirm.action)}
                style={{ padding: '6px 20px', borderRadius: 6, border: `1px solid ${confirm.action === 'halt' ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)'}`, background: confirm.action === 'halt' ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)', color: confirm.action === 'halt' ? 'var(--loss)' : 'var(--win)', cursor: 'pointer', fontSize: 13 }}>
                {confirm.action === 'halt' ? 'Trigger halt' : 'Clear halt'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
