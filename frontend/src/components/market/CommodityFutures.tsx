import React from 'react';
import { QuoteData } from '../../types/market';
import './CommodityFutures.css';
import { priceText, changeText, direction } from './quoteFormat';

interface CommodityFuturesProps {
  commodities: QuoteData[];
}

const CommodityFutures: React.FC<CommodityFuturesProps> = ({ commodities }) => {
  return (
    <div className="commodity-futures-card">
      <h2 className="section-title">Commodity Futures</h2>
      <div className="commodities-grid">
        {commodities.map((commodity) => (
          <div key={commodity.symbol} className="commodity-item">
            <div className="commodity-header">
              <span className="commodity-name">{commodity.name}</span>
              <span className="commodity-symbol">{commodity.symbol}</span>
            </div>
            <div className="commodity-data">
              <div className="commodity-price-section">
                <span className="commodity-label">Last</span>
                <span className="commodity-price">${priceText(commodity.last_price)}</span>
              </div>
              <div className="commodity-change-section">
                <span className="commodity-label">Change</span>
                <span className={`commodity-change ${direction(commodity.change)}`}>
                  {changeText(commodity.change, commodity.change_percent)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CommodityFutures;
