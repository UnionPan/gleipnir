import React, { useEffect, useState } from 'react';
import './App.css';
import HomePage from './pages/HomePage';
import CalibrationPage from './pages/CalibrationPage';
import HedgingLabPage from './pages/HedgingLabPage';
import LearningPage from './pages/LearningPage';
import { useTheme } from './contexts/ThemeContext';
import { BUNDLES_EVENT, unseenBundle } from './services/labBundles';

type Page = 'market' | 'calibration' | 'lab' | 'learning';

interface MenuItem {
  id: Page;
  label: string;
}

const menuItems: MenuItem[] = [
  { id: 'market', label: 'Market' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'lab', label: 'Hedging Lab' },
  { id: 'learning', label: 'Learned Hedgers' },
];

/* Logo: a delta glyph with a triangular counter */
const Logo = () => (
  <svg width="22" height="22" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Bold A/delta with triangular counter cutout */}
    <path d="M16 4 L27.5 26 L4.5 26 Z M16 15.5 L10.5 23 L21.5 23 Z" fill="url(#logoGrad)" fillRule="evenodd" />
    {/* Left-face highlight for depth */}
    <path d="M16 4 L4.5 26" stroke="rgba(255,255,255,0.12)" strokeWidth="0.8" />
    <defs>
      <linearGradient id="logoGrad" x1="5" y1="26" x2="27" y2="4">
        <stop offset="0%" stopColor="#06b6d4" />
        <stop offset="60%" stopColor="#6366f1" />
        <stop offset="100%" stopColor="#a78bfa" />
      </linearGradient>
    </defs>
  </svg>
);

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('market');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [labHasNew, setLabHasNew] = useState(() => unseenBundle() !== null);

  useEffect(() => {
    const refresh = () => setLabHasNew(unseenBundle() !== null);
    window.addEventListener(BUNDLES_EVENT, refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener(BUNDLES_EVENT, refresh);
      window.removeEventListener('storage', refresh);
    };
  }, []);
  const { theme, toggleTheme } = useTheme();

  const handlePageChange = (page: Page) => {
    setCurrentPage(page);
    setMobileMenuOpen(false);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'market':
        return <HomePage />;
      case 'calibration':
        return <CalibrationPage />;
      case 'lab':
        return <HedgingLabPage />;
      case 'learning':
        return <LearningPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="App">
      {/* Top Navigation */}
      <nav className="top-nav">
        <button type="button" className="nav-left" onClick={() => handlePageChange('market')} aria-label="Gleipnir home">
          <Logo />
          <span className="nav-brand">Gleip<span className="nav-brand-light">nir</span></span>
        </button>

        <div className="nav-center">
          <div className="nav-tabs">
            {menuItems.map((item) => (
              <button
                key={item.id}
                className={`nav-tab ${currentPage === item.id ? 'active' : ''}`}
                onClick={() => handlePageChange(item.id)}
              >
                {item.label}
                {item.id === 'lab' && labHasNew && currentPage !== 'lab' && (
                  <span className="nav-tab-new" title="A calibration is waiting in the Hedging Lab">
                    <span className="sr-only">(new calibration waiting)</span>
                  </span>
                )}
                {currentPage === item.id && <span className="nav-tab-indicator" />}
              </button>
            ))}
          </div>
        </div>

        <div className="nav-right">
          <div className="nav-status">
            <span className="status-dot" />
            <span className="status-text">Live</span>
          </div>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <circle cx="12" cy="12" r="5" />
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
        </div>

        {/* Mobile hamburger */}
        <button
          type="button"
          className="mobile-menu-button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={mobileMenuOpen}
        >
          <div className="hamburger">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </button>
      </nav>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <>
          <div className="mobile-menu-overlay" onClick={() => setMobileMenuOpen(false)}></div>
          <div className="mobile-menu">
            <div className="mobile-menu-header">
              <div className="mobile-menu-brand">
                <Logo />
                <h1 className="sidebar-title" onClick={() => handlePageChange('market')}>Gleipnir</h1>
              </div>
              <button type="button" className="close-button" onClick={() => setMobileMenuOpen(false)} aria-label="Close menu">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
            <nav className="mobile-nav">
              {menuItems.map((item) => (
                <button
                  key={item.id}
                  className={`mobile-nav-item ${currentPage === item.id ? 'active' : ''}`}
                  onClick={() => handlePageChange(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </>
      )}

      <main className="main-content">{renderPage()}</main>
    </div>
  );
}

export default App;
