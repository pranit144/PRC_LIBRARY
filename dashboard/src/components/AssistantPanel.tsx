import { useState } from 'react';
import { api } from '../lib/api';

interface Turn {
  question: string;
  answer: string;
}

const SUGGESTIONS = [
  'Why is validation loss increasing?',
  'Should I stop training?',
  'Which epoch was best?',
  'What should I try next?',
];

export function AssistantPanel({ runId }: { runId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  async function ask(question: string) {
    if (!question.trim() || loading) return;
    setLoading(true);
    setInput('');
    try {
      const res = await api.askAssistant(runId, question);
      setTurns((t) => [...t, { question, answer: res.answer }]);
    } catch (e) {
      setTurns((t) => [...t, { question, answer: 'Could not reach the assistant right now.' }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflowY: 'auto', marginBottom: 12, minHeight: 80 }}>
        {turns.length === 0 && (
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 10 }}>
              Ask about this run's metrics, anomalies, or forecast.
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  style={{
                    background: 'var(--panel-raised)',
                    border: '1px solid var(--line)',
                    color: 'var(--text-secondary)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 11,
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>{t.question}</div>
            <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.5 }}>{t.answer}</div>
          </div>
        ))}
        {loading && <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>thinking…</div>}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        style={{ display: 'flex', gap: 8 }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this run…"
          style={{
            flex: 1,
            background: 'var(--panel-raised)',
            border: '1px solid var(--line)',
            borderRadius: 6,
            padding: '8px 10px',
            color: 'var(--text-primary)',
            fontSize: 13,
            fontFamily: 'inherit',
          }}
        />
        <button
          type="submit"
          style={{
            background: 'var(--observed-dim)',
            border: '1px solid var(--observed)',
            color: 'var(--observed)',
            borderRadius: 6,
            padding: '8px 14px',
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          Ask
        </button>
      </form>
    </div>
  );
}
