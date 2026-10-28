import React from 'react';
import { QuoteData } from '../../types/market';
import './Magnificent7.css';
import { priceText, percentText, direction } from './quoteFormat';

interface Magnificent7Props {
  stocks: QuoteData[];
}

const Magnificent7: React.FC<Magnificent7Props> = ({ stocks }) => {
  return (
    <div className="magnificent7-card">
      <h2 className="section-title">Magnificent 7</h2>
      <div className="magnificent7-grid">
        {stocks.map((stock) => (
          <div key={stock.symbol} className="stock-item">
            <div className="stock-header">
              <span className="stock-symbol">{stock.symbol}</span>
              <span className={`stock-change ${direction(stock.change_percent)}`}>
                {percentText(stock.change_percent)}
              </span>
            </div>
            <div className="stock-name">{stock.name}</div>
            <div className="stock-price">${priceText(stock.last_price)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Magnificent7;
