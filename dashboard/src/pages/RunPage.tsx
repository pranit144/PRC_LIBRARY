import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api, connectRunSocket, type Run, type MetricPoint, type Anomaly, type Forecast } from '../lib/api';
import { Pulse } from '../components/Pulse';
import { MetricCards } from '../components/MetricCards';
import { TrainingChart } from '../components/TrainingChart';
import { Timeline, buildTimeline } from '../components/Timeline';
import { ForecastPanel } from '../components/ForecastPanel';
import { ModelHealth } from '../components/ModelHealth';
import { AssistantPanel } from '../components/AssistantPanel';

function Panel({ title, children, style }: { title: string; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        background: 'var(--panel)',
        border: '1px solid var(--line)',
        borderRadius: 10,
        padding: 20,
        ...style,
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 14 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

export function RunPage() {
  const { runId = '' } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    // Reflect run status in the browser tab title, so it's visibly
    // "Live Monitoring" for a running run even before the first
    // metric arrives, and reverts once the run finishes.
    if (!run) return;
    document.title = run.status === 'running'
      ? `\u25CF Live Monitoring — ${run.run_name}`
      : `${run.run_name} — prc`;
    return () => {
      document.title = 'prc';
    };
  }, [run?.status, run?.run_name]);

  async function refresh() {
    try {
      const [r, m] = await Promise.all([api.getRun(runId), api.getMetrics(runId)]);
      setRun(r);
      setMetrics(m);
      setAnomalies(r.anomalies || []);
      setForecast(r.forecast || null);
      setError(null);
    } catch (e: any) {
      setError(e.message || 'Could not load run.');
    }
  }

  useEffect(() => {
    refresh();
    pollRef.current = window.setInterval(refresh, 4000);

    const disconnect = connectRunSocket(runId, (msg) => {
      if (msg.event === 'metric') {
        setMetrics((prev) => [...prev, { step: msg.step, epoch: msg.epoch, timestamp: msg.timestamp, ...msg.data.metrics }]);
      } else if (msg.event === 'anomaly') {
        setAnomalies((prev) => [...prev, msg.data]);
      } else if (msg.event === 'run_finished') {
        refresh();
      }
    });

    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  useEffect(() => {
    // Pull a fresh forecast periodically as new metrics arrive.
    if (metrics.length >= 5) {
      api.getForecast(runId, 'val_loss').then(setForecast).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metrics.length]);

  if (error) {
    return (
      <div style={{ padding: 48, color: 'var(--bad)' }}>
        {error} — <Link to="/" style={{ textDecoration: 'underline' }}>back to runs</Link>
      </div>
    );
  }

  if (!run) {
    return <div style={{ padding: 48, color: 'var(--text-faint)' }}>Loading…</div>;
  }

  const last = metrics[metrics.length - 1] || {};
  const valLossHistory = metrics.map((m) => m.val_loss).filter((v) => v !== undefined);
  const isLive = run.status === 'running';

  return (
    <div style={{ maxWidth: 1180, margin: '0 auto', padding: '32px 24px 64px' }}>
      <Link to="/" style={{ fontSize: 12, color: 'var(--text-faint)' }}>&larr; all runs</Link>

      {/* -- header -------------------------------------------------- */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', margin: '10px 0 28px', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 4 }}>
            {run.project}
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 10 }}>
            {run.run_name}
            {isLive && (
              <span
                className="mono"
                style={{ fontSize: 10, fontWeight: 700, color: 'var(--observed)', border: '1px solid var(--observed)', borderRadius: 4, padding: '3px 7px' }}
              >
                <span style={{ display: 'inline-block', width: 5, height: 5, borderRadius: '50%', background: 'var(--observed)', marginRight: 5 }} />
                LIVE
              </span>
            )}
          </h1>
          <div className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>
            Epoch {run.last_epoch} · Step {run.last_step.toLocaleString()}
            {forecast?.status === 'ok' && forecast.estimated_convergence_step != null && (
              <> · ETA step {forecast.estimated_convergence_step.toLocaleString()}</>
            )}
          </div>
        </div>

        {valLossHistory.length > 1 && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              val loss pulse
            </div>
            <Pulse values={valLossHistory.slice(-40)} live={isLive} />
          </div>
        )}
      </div>

      {/* -- metric cards -------------------------------------------------- */}
      <div style={{ marginBottom: 20 }}>
        <MetricCards metrics={last as any} />
      </div>

      {/* -- main grid -------------------------------------------------- */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, marginBottom: 20 }}>
        <Panel title="Training / validation loss">
          <TrainingChart metrics={metrics} forecast={forecast} />
        </Panel>
        <Panel title="Forecast">
          <ForecastPanel forecast={forecast} />
        </Panel>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 320px', gap: 20 }}>
        <Panel title="Training timeline">
          <Timeline entries={buildTimeline(anomalies, run)} />
        </Panel>
        <Panel title="Model health">
          <ModelHealth anomalies={anomalies} />
        </Panel>
        <Panel title="Assistant" style={{ minHeight: 260, display: 'flex', flexDirection: 'column' }}>
          <AssistantPanel runId={runId} />
        </Panel>
      </div>
    </div>
  );
}
