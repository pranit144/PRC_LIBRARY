import type { Forecast } from '../lib/api';

export function ForecastPanel({ forecast }: { forecast: Forecast | null | undefined }) {
  if (!forecast || forecast.status === 'insufficient_data') {
    return (
      <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>
        {forecast?.message || 'Insufficient data for reliable forecast.'}
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
        <span
          className="mono"
          style={{ fontSize: 26, fontWeight: 700, color: 'var(--predicted)' }}
        >
          {forecast.predicted_final}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          projected final {forecast.metric}
        </span>
      </div>
      <div className="mono" style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 12 }}>
        range {forecast.lower_bound}–{forecast.upper_bound}
      </div>

      <div style={{ display: 'flex', gap: 20, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-faint)' }}>
            Confidence
          </div>
          <div className="mono" style={{ fontSize: 15, color: 'var(--text-primary)' }}>
            {Math.round((forecast.confidence ?? 0) * 100)}%
          </div>
        </div>
        {forecast.estimated_convergence_step != null && (
          <div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-faint)' }}>
              Est. convergence
            </div>
            <div className="mono" style={{ fontSize: 15, color: 'var(--text-primary)' }}>
              step {forecast.estimated_convergence_step}
            </div>
          </div>
        )}
      </div>

      {/* confidence bar, rendered with the predicted color to reinforce that this is an estimate */}
      <div style={{ height: 4, background: 'var(--line-soft)', borderRadius: 2, overflow: 'hidden', marginBottom: 10 }}>
        <div
          style={{
            height: '100%',
            width: `${Math.round((forecast.confidence ?? 0) * 100)}%`,
            background: 'var(--predicted)',
          }}
        />
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-faint)', lineHeight: 1.4 }}>
        {forecast.disclaimer || 'Estimate only, not a guarantee.'}
      </div>
    </div>
  );
}
