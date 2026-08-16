import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ProjectsPage } from './pages/ProjectsPage';

// RunPage pulls in recharts, which is the single heaviest dependency in
// this app. The projects list page never needs it, so we lazy-load
// RunPage instead of bundling it into the initial page load.
const RunPage = lazy(() => import('./pages/RunPage').then((m) => ({ default: m.RunPage })));

function PageFallback() {
  return (
    <div style={{ padding: 48, color: 'var(--text-faint)', fontSize: 13 }}>
      Loading…
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/runs/:runId" element={<RunPage />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
