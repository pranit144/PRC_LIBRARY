import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import type { MetricPoint, Forecast } from '../lib/api';

interface TrainingChartProps {
  metrics: MetricPoint[];
  forecast?: Forecast | null;
}

export function TrainingChart({ metrics, forecast }: TrainingChartProps) {
  const lastStep = metrics.length ? metrics[metrics.length - 1].step : 0;

  // Build a forecast "tail": a couple of synthetic points continuing past
  // the last observed step, so the dashed line visually extends the curve.
  const forecastTail =
    forecast && forecast.status === 'ok' && forecast.predicted_final !== undefined
      ? [
          { step: lastStep, val_loss_forecast: forecast.current },
          {
            step: lastStep + (forecast.estimated_convergence_step ? forecast.estimated_convergence_step - lastStep : lastStep * 0.5 + 10),
            val_loss_forecast: forecast.predicted_final,
          },
        ]
      : [];

  const combined = [...metrics, ...forecastTail.slice(1)];

  return (
    <div style={{ width: '100%', height: 280 }}>
      <ResponsiveContainer>
        <ComposedChart data={combined} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="var(--line-soft)" vertical={false} />
          <XAxis
            dataKey="step"
            stroke="var(--text-faint)"
            tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={{ stroke: 'var(--line)' }}
          />
          <YAxis
            stroke="var(--text-faint)"
            tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--panel-raised)',
              border: '1px solid var(--line)',
              borderRadius: 6,
              fontSize: 12,
              fontFamily: 'var(--font-mono)',
            }}
            labelStyle={{ color: 'var(--text-secondary)' }}
          />
          {lastStep > 0 && (
            <ReferenceLine x={lastStep} stroke="var(--text-faint)" strokeDasharray="2 3" label={{ value: 'now', position: 'top', fill: 'var(--text-faint)', fontSize: 10 }} />
          )}
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            name="train loss"
          />
          <Line
            type="monotone"
            dataKey="val_loss"
            stroke="var(--observed)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="val loss (observed)"
          />
          {forecastTail.length > 0 && (
            <Line
              type="monotone"
              dataKey="val_loss_forecast"
              stroke="var(--predicted)"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
              name="val loss (forecast)"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
