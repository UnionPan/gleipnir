import React from 'react';
import { QuoteData } from '../../types/market';
import './MarketIndices.css';
import { priceText, changeText, direction } from './quoteFormat';

interface MarketIndicesProps {
  indices: QuoteData[];
}

const MarketIndices: React.FC<MarketIndicesProps> = ({ indices }) => {
  return (
    <div className="market-indices-card">
      <h2 className="section-title">US Equities</h2>
      <div className="indices-list">
        {indices.map((quote) => (
          <div key={quote.symbol} className="index-item">
            <div className="index-header">
              <span className="index-name">{quote.name}</span>
              <span className="index-symbol">{quote.symbol}</span>
            </div>
            <div className="index-data">
              <span className="index-price">${priceText(quote.last_price)}</span>
              <span className={`index-change ${direction(quote.change)}`}>
                {changeText(quote.change, quote.change_percent)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MarketIndices;
