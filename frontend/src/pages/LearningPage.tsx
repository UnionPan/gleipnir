import React, { useEffect, useState } from 'react';
import './HedgingLabPage.css';
import './LearningPage.css';
import { useTheme } from '../contexts/ThemeContext';
import { fetchLearningDemos, LearningDemos } from '../services/rlApi';
import DeepHedgingSection from '../components/learning/DeepHedgingSection';
import CliquetSection from '../components/learning/CliquetSection';

/** Two trained controllers, evaluated offline on fixed paths: the page only reads their results. */
const LearningPage: React.FC = () => {
  const { theme } = useTheme();
  const [demos, setDemos] = useState<LearningDemos | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        <h1>Learning to Hedge</h1>
        <p className="page-subtitle">
          Two controllers trained offline on a GPU and evaluated on fixed paths: a deep hedger that learns when
          options are worth their cost, and an agent that must remember a contract&apos;s history it cannot see.
        </p>
      </header>
      {error && <div className="lab-error" role="alert">{error}</div>}
      {!demos && !error && <p className="lab-empty">Loading the trained hedgers…</p>}
      {demos?.deep_hedging && <DeepHedgingSection demo={demos.deep_hedging} theme={theme} />}
      {demos?.cliquet_ais && <CliquetSection demo={demos.cliquet_ais} theme={theme} />}
    </div>
  );
};

export default LearningPage;
