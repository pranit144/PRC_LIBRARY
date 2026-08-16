import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, type Project, type Run } from '../lib/api';

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [runsByProject, setRunsByProject] = useState<Record<string, Run[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const projs = await api.listProjects();
        if (cancelled) return;
        setProjects(projs);
        const entries = await Promise.all(
          projs.map(async (p) => [p.name, await api.listRuns(p.name)] as const)
        );
        if (cancelled) return;
        setRunsByProject(Object.fromEntries(entries));
      } catch (e: any) {
        if (!cancelled) setError(e.message || 'Could not reach the prc server.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '48px 24px' }}>
      <header style={{ marginBottom: 40 }}>
        <div style={{ fontSize: 12, letterSpacing: '0.12em', color: 'var(--observed)', marginBottom: 6, textTransform: 'uppercase' }}>
          prc
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
          See the training. Understand the model. Forecast what comes next.
        </h1>
      </header>

      {error && (
        <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 24 }}>
          {error} — is the prc server running at the configured URL?
        </div>
      )}

      {!error && loading && <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>Loading…</div>}

      {!error && !loading && projects.length === 0 && (
        <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>
          No runs yet. Start one with the prc SDK — see <code>examples/mnist</code> for a working example.
        </div>
      )}

      <div style={{ display: 'grid', gap: 28 }}>
        {projects.map((p) => (
          <section key={p.name}>
            <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {p.name}
            </h2>
            <div style={{ display: 'grid', gap: 8 }}>
              {(runsByProject[p.name] || []).map((r) => (
                <Link
                  key={r.run_id}
                  to={`/runs/${r.run_id}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'var(--panel)',
                    border: '1px solid var(--line)',
                    borderRadius: 8,
                    padding: '12px 16px',
                  }}
                >
                  <div>
                    <div style={{ fontSize: 14, color: 'var(--text-primary)' }}>{r.run_name}</div>
                    <div className="mono" style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 2 }}>
                      {r.run_id} · epoch {r.last_epoch} · step {r.last_step}
                    </div>
                  </div>
                  <StatusBadge status={r.status} />
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Run['status'] }) {
  const map = {
    running: { color: 'var(--observed)', label: 'LIVE' },
    completed: { color: 'var(--good)', label: 'DONE' },
    failed: { color: 'var(--bad)', label: 'FAILED' },
  } as const;
  const s = map[status] || map.completed;
  return (
    <span
      className="mono"
      style={{
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.08em',
        color: s.color,
        border: `1px solid ${s.color}`,
        borderRadius: 4,
        padding: '3px 7px',
      }}
    >
      {status === 'running' && (
        <span
          style={{
            display: 'inline-block',
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: s.color,
            marginRight: 5,
          }}
        />
      )}
      {s.label}
    </span>
  );
}
