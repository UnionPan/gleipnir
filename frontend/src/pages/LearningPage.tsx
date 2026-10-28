import React, { useEffect, useState } from 'react';
import './HedgingLabPage.css';
import './LearningPage.css';
import { useTheme } from '../contexts/ThemeContext';
import { fetchLearningDemos, LearningDemos } from '../services/rlApi';
import DeepHedgingSection from '../components/learning/DeepHedgingSection';
import CliquetSection from '../components/learning/CliquetSection';

type Tab = 'deep' | 'cliquet';

const TABS: { key: Tab; title: string; question: string }[] = [
  { key: 'deep', title: 'Deep hedger', question: 'Are options worth trading to hedge an option?' },
  { key: 'cliquet', title: 'Learned memory', question: 'Can an agent hedge a contract whose state it cannot see?' },
];

/** Two controllers trained offline and evaluated on fixed paths: the page only reads their results. */
const LearningPage: React.FC = () => {
  const { theme } = useTheme();
  const [demos, setDemos] = useState<LearningDemos | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('deep');

  useEffect(() => {
    let live = true;
    fetchLearningDemos()
      .then((d) => { if (live) setDemos(d); })
      .catch((e: Error) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, []);

  return (
    <div className="lab-page learn-page">
      <header className="page-header">
        <h1>Learned Hedgers</h1>
        <p className="page-subtitle">
          Controllers trained offline on a GPU, replayed on simulated markets next to the classical hedges they would replace.
        </p>
      </header>
      <div className="learn-tabs" role="tablist" aria-label="Demos">
        {TABS.map((t) => (
          <button key={t.key} type="button" role="tab" id={`learn-tab-${t.key}`} aria-selected={tab === t.key}
            aria-controls={`learn-panel-${t.key}`} className={`learn-tab-button ${tab === t.key ? 'active' : ''}`}
            onClick={() => setTab(t.key)}>
            <strong>{t.title}</strong>
            <span>{t.question}</span>
          </button>
        ))}
      </div>
      {error && <div className="lab-error" role="alert">{error}</div>}
      {!demos && !error && <p className="lab-empty">Loading the trained hedgers…</p>}
      <div role="tabpanel" id={`learn-panel-${tab}`} aria-labelledby={`learn-tab-${tab}`}>
        {tab === 'deep' && demos?.deep_hedging && <DeepHedgingSection demo={demos.deep_hedging} theme={theme} />}
        {tab === 'cliquet' && demos?.cliquet_ais && <CliquetSection demo={demos.cliquet_ais} theme={theme} />}
      </div>
    </div>
  );
};

export default LearningPage;
