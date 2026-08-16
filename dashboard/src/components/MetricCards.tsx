interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  accent?: 'observed' | 'neutral';
}

function MetricCard({ label, value, unit, accent = 'neutral' }: MetricCardProps) {
  return (
    <div
      style={{
        background: 'var(--panel)',
        border: '1px solid var(--line)',
        borderRadius: 8,
        padding: '14px 16px',
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 8 }}>
        {label}
      </div>
      <div
        className="mono"
        style={{
          fontSize: 22,
          fontWeight: 600,
          color: accent === 'observed' ? 'var(--observed)' : 'var(--text-primary)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {value}
        {unit && <span style={{ fontSize: 13, color: 'var(--text-secondary)', marginLeft: 4 }}>{unit}</span>}
      </div>
    </div>
  );
}

export interface MetricCardsProps {
  metrics: {
    train_loss?: number;
    val_loss?: number;
    accuracy?: number;
    learning_rate?: number;
    gpu_memory_utilization_pct?: number;
    gpu_name?: string;
  };
}

function fmt(n: number | undefined, digits = 4): string {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  return n.toFixed(digits);
}

export function MetricCards({ metrics }: MetricCardsProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: 12,
      }}
    >
      <MetricCard label="Train loss" value={fmt(metrics.train_loss)} accent="observed" />
      <MetricCard label="Val loss" value={fmt(metrics.val_loss)} accent="observed" />
      <MetricCard label="Accuracy" value={metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}` : '—'} unit="%" />
      <MetricCard label="Learning rate" value={metrics.learning_rate !== undefined ? metrics.learning_rate.toExponential(2) : '—'} />
      <MetricCard
        label="GPU memory"
        value={metrics.gpu_memory_utilization_pct !== undefined ? metrics.gpu_memory_utilization_pct.toFixed(0) : '—'}
        unit="%"
      />
    </div>
  );
}
