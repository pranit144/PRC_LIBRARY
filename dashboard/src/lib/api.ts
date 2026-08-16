// Default to same-origin: when the dashboard is served by the prc
// server itself (single-port mode - see server/main.py), this just
// works no matter what host/URL you're on, including proxied URLs like
// Colab's kernel.proxyPort() links. Override with VITE_PRC_SERVER_URL
// only if the dashboard is running separately from the API (e.g. local
// `npm run dev` on :5173 talking to the API on :8000).
const BASE_URL = import.meta.env.VITE_PRC_SERVER_URL || window.location.origin;
const API_BASE = `${BASE_URL}/api`;
const WS_BASE_URL = BASE_URL.replace(/^http/, 'ws');

export interface Project {
  name: string;
  created_at: string;
}

export interface Run {
  run_id: string;
  project: string;
  run_name: string;
  config: Record<string, any>;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  finished_at: string | null;
  last_step: number;
  last_epoch: number;
  anomalies?: Anomaly[];
  forecast?: Forecast | null;
}

export interface Anomaly {
  type: string;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  message: string;
}

export interface Forecast {
  metric: string;
  status: 'ok' | 'insufficient_data';
  message?: string;
  current?: number;
  predicted_final?: number;
  lower_bound?: number;
  upper_bound?: number;
  confidence?: number;
  estimated_convergence_step?: number | null;
  disclaimer?: string;
}

export interface MetricPoint {
  step: number;
  epoch: number;
  timestamp: string;
  [metric: string]: any;
}

export interface PrcEvent {
  run_id: string;
  event: string;
  schema_version: number;
  timestamp: string;
  step: number | null;
  epoch: number | null;
  data: Record<string, any>;
}

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    throw new Error(`prc API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export const api = {
  health: () => req<{ status: string }>('/health'),
  listProjects: () => req<Project[]>('/projects'),
  getProject: (project: string) => req<Project & { runs: Run[] }>(`/projects/${encodeURIComponent(project)}`),
  listRuns: (project?: string) =>
    req<Run[]>(`/runs${project ? `?project=${encodeURIComponent(project)}` : ''}`),
  getRun: (runId: string) => req<Run>(`/runs/${encodeURIComponent(runId)}`),
  getEvents: (runId: string) => req<PrcEvent[]>(`/runs/${encodeURIComponent(runId)}/events`),
  getMetrics: (runId: string) => req<MetricPoint[]>(`/runs/${encodeURIComponent(runId)}/metrics`),
  getForecast: (runId: string, metric = 'val_loss') =>
    req<Forecast>(`/runs/${encodeURIComponent(runId)}/forecast?metric=${encodeURIComponent(metric)}`),
  askAssistant: (runId: string, question: string) =>
    req<{ question: string; answer: string }>(
      `/runs/${encodeURIComponent(runId)}/assistant?question=${encodeURIComponent(question)}`
    ),
};

export function connectRunSocket(runId: string, onMessage: (msg: any) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retryTimer: number | undefined;

  const connect = () => {
    if (closed) return;
    ws = new WebSocket(`${WS_BASE_URL}/api/ws/runs/${encodeURIComponent(runId)}`);
    ws.onmessage = (evt) => {
      try {
        onMessage(JSON.parse(evt.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => {
      if (!closed) {
        retryTimer = window.setTimeout(connect, 2000);
      }
    };
    ws.onerror = () => {
      ws?.close();
    };
  };

  connect();

  return () => {
    closed = true;
    if (retryTimer) window.clearTimeout(retryTimer);
    ws?.close();
  };
}

export { BASE_URL };
