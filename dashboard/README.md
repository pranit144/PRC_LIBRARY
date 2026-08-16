# prc dashboard

React + TypeScript dashboard for prc. Talks to the prc server over REST
for initial loads and WebSocket for live updates.

## Design

Built as an instrument panel, not a generic SaaS admin dashboard:

- **Amber (`--observed`)** marks real, measured data.
- **Cyan (`--predicted`)** marks forecasted/predicted data — this
  distinction is used consistently in the chart, forecast panel, and
  anywhere a projection appears, so observed and predicted values are
  never visually ambiguous.
- **Monospace numerals** throughout, since this is a readout of live
  instrumentation, not prose.
- The header's "pulse" sparkline is the signature element — a live
  heartbeat-style trace of validation loss, echoing the idea that a
  training run is a living process to be monitored.

## Setup

**Production / single-port mode (recommended):** the dashboard is built
and served directly by the prc server — see the root README's Quick
Start. No separate dev server or env var needed; API calls default to
same-origin (`window.location.origin`), which is what makes the
Colab/Kaggle proxy links work.

**Dev mode (hot reload, separate process):**

```bash
npm install
npm run dev
```

By default this still targets same-origin, which won't be correct when
running as a separate dev server. Point it at the API explicitly:

```bash
VITE_PRC_SERVER_URL=http://localhost:8000 npm run dev
```

## Structure

```
src/
├── lib/api.ts          REST client + WebSocket connection helper
├── components/
│   ├── Pulse.tsx        signature sparkline
│   ├── MetricCards.tsx  live metric readouts
│   ├── TrainingChart.tsx  loss/accuracy curves + forecast overlay
│   ├── Timeline.tsx     chronological event feed
│   ├── ForecastPanel.tsx
│   ├── ModelHealth.tsx  progressive-disclosure health summary
│   └── AssistantPanel.tsx
└── pages/
    ├── ProjectsPage.tsx  list of projects/runs
    └── RunPage.tsx       main run dashboard
```

## Build

```bash
npm run build
```
