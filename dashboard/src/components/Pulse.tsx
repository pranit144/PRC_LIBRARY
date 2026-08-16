import { useMemo } from 'react';

interface PulseProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  live?: boolean;
}

/**
 * A small heartbeat-style sparkline used in the run header. It's the
 * dashboard's signature element: training is a living process, and this
 * is its pulse. The trailing dot glows and pulses while the run is live.
 */
export function Pulse({ values, width = 160, height = 36, color = 'var(--observed)', live = false }: PulseProps) {
  const path = useMemo(() => {
    if (values.length < 2) return '';
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const stepX = width / (values.length - 1);
    return values
      .map((v, i) => {
        const x = i * stepX;
        const y = height - ((v - min) / range) * (height - 6) - 3;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [values, width, height]);

  const last = values[values.length - 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const lastY = values.length ? height - ((last - min) / range) * (height - 6) - 3 : height / 2;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
      <path d={path} fill="none" stroke={color} strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
      {values.length > 0 && (
        <circle cx={width} cy={lastY} r={3} fill={color}>
          {live && (
            <animate attributeName="opacity" values="1;0.35;1" dur="1.6s" repeatCount="indefinite" />
          )}
        </circle>
      )}
    </svg>
  );
}
