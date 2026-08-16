import type { Anomaly } from '../lib/api';

interface TimelineEntry {
  step: number | null;
  epoch: number | null;
  label: string;
  detail?: string;
  tone: 'neutral' | 'good' | 'warn' | 'bad';
}

function severityTone(severity: Anomaly['severity']): 'warn' | 'bad' {
  return severity === 'high' ? 'bad' : 'warn';
}

const toneColor: Record<TimelineEntry['tone'], string> = {
  neutral: 'var(--text-faint)',
  good: 'var(--good)',
  warn: 'var(--warn)',
  bad: 'var(--bad)',
};

export function buildTimeline(anomalies: Anomaly[], run: { started_at: string; finished_at: string | null; last_epoch: number }): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  entries.push({ step: null, epoch: null, label: 'Training started', tone: 'neutral' });

  for (const a of anomalies) {
    entries.push({
      step: null,
      epoch: null,
      label: a.type.replace(/_/g, ' '),
      detail: a.message,
      tone: severityTone(a.severity),
    });
  }

  if (run.finished_at) {
    entries.push({ step: null, epoch: run.last_epoch, label: 'Run finished', tone: 'good' });
  }

  return entries;
}

export function Timeline({ entries }: { entries: TimelineEntry[] }) {
  if (entries.length === 0) {
    return <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>No events yet.</div>;
  }

  return (
    <div style={{ position: 'relative', paddingLeft: 18 }}>
      <div style={{ position: 'absolute', left: 4, top: 6, bottom: 6, width: 1, background: 'var(--line)' }} />
      {entries.map((e, i) => (
        <div key={i} style={{ position: 'relative', paddingBottom: i === entries.length - 1 ? 0 : 16 }}>
          <div
            style={{
              position: 'absolute',
              left: -18 + 1,
              top: 4,
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: toneColor[e.tone],
              boxShadow: e.tone !== 'neutral' ? `0 0 0 3px ${toneColor[e.tone]}22` : 'none',
            }}
          />
          <div style={{ fontSize: 13, color: 'var(--text-primary)', textTransform: 'capitalize' }}>{e.label}</div>
          {e.detail && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2, lineHeight: 1.4 }}>{e.detail}</div>
          )}
        </div>
      ))}
    </div>
  );
}
