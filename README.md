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

---

## Data Acquisition
Retrieves raw market data and the 10 most recent news articles for Nasdaq-100 symbols and calculates sentiment score.

**Script**

[scripts/01data_acquisition.py](scripts/01data_acquisition.py)

Pulls **1‑minute** bars from **2020‑01‑01 to 2025‑11‑21** and writes `symbol.parquet` files to  
`data/raw/Prices_1m_adj`.

- `columns`: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trade_count`,`vwap`, `symbol`   

Bar data AAPL example:

<img width="1462" height="906" alt="image" src="https://github.com/user-attachments/assets/84685933-70b6-4efa-bc9e-2b6509b03899" />

[scripts/news_sentiment_score.py](scripts/news_sentiment_score.py)

Calculates the sentiment score of recent news regarding each stock and creates `data/processed/nasdaq_news_with_sentiment.parquet` file.

Sentiment score AAPL example:

<img width="1266" height="697" alt="image" src="https://github.com/user-attachments/assets/a7f688a7-1021-47f4-aa78-4732f7ee627c" />

**API**

#### Alpaca Market Data API
Ruft historische 1-Minuten-Kerzendaten für NASDAQ-100 Symbole und QQQ über die Alpaca Data API ab. Die Daten werden im Parquet-Format unter `data/raw/Prices_1m_adj` gespeichert.


#### Parameter
`symbol_or_symbols`: Liste von Tickern
`timeframe`: 1Min (1-Minuten Bars)
`start`: Start-Datum 2025-01-01
`end`: End-Datum 2025-11-21
`adjustment`: adjustiert für Splits & Dividenden


#### Alpaca Trading API
Filtern nach den Minuten, die innerhalb der regulären Handelszeiten des offiziellen US-Handelskalender liegen.


#### Parameter
`start`: Start-Datum 
`end`: End-Datum


#### Yahoo Finance API
Ruft die letzten 10 Nachrichtenartikel für NASDAQ-bezogene Symbole über die Yahoo Finance API ab. Die Daten werden in `data/raw/News_raw` gespeichert.


#### Parameter 
`ticker`: erstellt ein Ticker-Objekt
`news_items`: gibt eine Liste von News-Einträgen


#### FinBERT API
Berechnet den sentiment score der letzten 10 Nachrichtenartikel. Die Daten werden in `data/processed/nasdaq_news_with_sentiment.parquet` gespeichert.

#### Parameter
`text`: (Headline + Summary
`return_tensors`:pt (Ausgabe als PyTorch-Tensors)
`truncation`:True (Text wird auf max_length abgeschnitten)
`max_length`:512 (maximale Tokenlänge)

