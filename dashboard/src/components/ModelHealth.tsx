import { useState } from 'react';
import type { Anomaly } from '../lib/api';

interface HealthDim {
  key: string;
  label: string;
  status: 'good' | 'warn' | 'bad';
  note: string;
}

function computeHealth(anomalies: Anomaly[]): HealthDim[] {
  const has = (t: string) => anomalies.find((a) => a.type === t);

  const overfit = has('overfitting');
  const gradIssue = has('vanishing_gradients') || has('exploding_gradients');
  const unstable = has('unstable_training');
  const plateau = has('plateau');

  return [
    {
      key: 'gradients',
      label: 'Gradient health',
      status: gradIssue ? 'bad' : 'good',
      note: gradIssue ? gradIssue!.message : 'Gradient magnitudes look within a normal range.',
    },
    {
      key: 'generalization',
      label: 'Generalization',
      status: overfit ? (overfit.severity === 'high' ? 'bad' : 'warn') : 'good',
      note: overfit ? overfit.message : 'Validation loss is tracking training loss.',
    },
    {
      key: 'stability',
      label: 'Training stability',
      status: unstable ? 'warn' : 'good',
      note: unstable ? unstable.message : 'Loss curve looks smooth, no large oscillations detected.',
    },
    {
      key: 'progress',
      label: 'Progress',
      status: plateau ? 'warn' : 'good',
      note: plateau ? plateau.message : 'Metrics are still improving.',
    },
  ];
}

const dot: Record<HealthDim['status'], string> = {
  good: 'var(--good)',
  warn: 'var(--warn)',
  bad: 'var(--bad)',
};

export function ModelHealth({ anomalies }: { anomalies: Anomaly[] }) {
  const [expanded, setExpanded] = useState(false);
  const dims = computeHealth(anomalies);
  const worst = dims.some((d) => d.status === 'bad') ? 'bad' : dims.some((d) => d.status === 'warn') ? 'warn' : 'good';

  const summary =
    worst === 'good'
      ? 'Model is learning normally.'
      : worst === 'warn'
      ? `⚠️ ${dims.find((d) => d.status !== 'good')?.note}`
      : `⚠️ ${dims.find((d) => d.status === 'bad')?.note}`;

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: expanded ? 14 : 0 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: dot[worst], flexShrink: 0 }} />
        <span style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.4 }}>{summary}</span>
      </div>

      <button
        onClick={() => setExpanded((e) => !e)}
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--text-faint)',
          fontSize: 11,
          padding: 0,
          marginTop: 8,
          textDecoration: 'underline',
          textUnderlineOffset: 3,
        }}
      >
        {expanded ? 'Hide details' : 'Show details'}
      </button>

      {expanded && (
        <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
          {dims.map((d) => (
            <div key={d.key} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: dot[d.status], flexShrink: 0, marginTop: 4 }} />
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-primary)' }}>{d.label}</div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>{d.note}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
