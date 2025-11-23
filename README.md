# USW_Trading

Problem Definition:
Target

Prediction of 60 minute trend direction for NASDAQ-100 stocks during regular US trading hours. For every NASDAQ-100 stock and every 1-minute bar between 2020-01-01 and 2025-11-21 we compute the target as the normalized linear regression slope of the stock’s future 60-minute close-price window, where the slope is normalized by the mean price in that window.


-Normalized intraday OHLCV (open, high, low, close, volume) at 1-minute frequency and corresponding log returns

-Normalized exponential moving averages (EMA) of price and volume over t = [15, 30, 
60, 120] minutes

-Stock-specific volatility features

-Normalized index features such as NASDAQ-100 index return and intraday volatility

-Sentiment scores from recent headlines and summaries per stock 



Procedure Overview:

-Collects for all NASDAQ-100 tickers 1-minute bars from 2020-01-01 to 2025-11-21 via Alpaca and yfinance and retrieves recent news per ticker to compute sentiment scores.  
-Engineers above features for each symbol for each minute.
-Predicts direction of trend over next 60 minutes (entry-network) using feed-forward neural network ([128, 64] hidden layers, dropout 0.1, ReLU activation).
-Use decision tree (depth=10) with embeddings (hidden layer with 64 neurons) to predict entry points with positive trend direction.
Implement trading strategy in Alpaca, that enters positions at predicted entry points and holds them for 60 minutes.



-We hope to find that combining technical indicators with news sentiment improves short-term trend prediction and leads to better risk-adjusted returns than a purely price-based baseline.
