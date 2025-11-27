# USW_Trading

### Problem Definition:
**Ziel**

Vorhersage der nächsten t=[15, 30, 60, 120] Minuten Trendrichtung für NASDAQ-100-Aktien während der regulären US-Handelszeiten. Für jede NASDAQ-100-Aktie und jede 1-Minuten-Kerze im Zeitraum vom 01.01.2020 bis 21.11.2025 berechnen wir die die erwartete Preisveränderung über die zukünftigen t Minuten der Aktie. Dabei werden Minuten-Daten und News genutzt, damit wiederkehrende Strukturen wie Volatilität oder newsgetriebene Bewegungen identifiziert werden.


**Input Features**

- Normalisierte Intraday-OHLCV-Daten (Open, High, Low, Close, Volume) mit 1-Minuten-Frequenz und den zugehörigen Log-Returns
- Normalisierte exponentielle gleitende Durchschnitte (EMA) von Preis und Volumen über t = [5, 15, 30, 60, 120] Minuten
- Aktien­spezifische Volatilitätsmerkmale
- Normalisierte Indexmerkmale wie NASDAQ-100-Indexrendite und Intraday-Volatilität
- Sentiment-Scores aus Überschriften und Zusammenfassungen pro Aktie


### Verfahrensübersicht:

- Sammelt für alle NASDAQ-100-Ticker 1-Minuten-Kursdaten für den Zeitraum vom 01.01.2020 bis 21.11.2025 und ruft Nachrichten pro Ticker ab, um Sentiment-Scores zu berechnen.
- Sagt die Richtung des Trends über nächste t Minuten vorraus mit einem Neural Network
- Verwendet einen Entscheidungsbaum, um Einstiege mit positiver Trendrichtung zu prognostizieren
- Implementiert eine Trading Strategie in Alpaca


**Wir hoffen zu zeigen, dass die Kombination technischer Indikatoren mit News-Sentiment die kurzfristige Trendvorhersage verbessert**

---

## Data Acquisition
Ruft rohe Marktdaten und Nachrichtenartikel für Nasdaq-100-Symbole ab und berechnet den Sentiment-Score.

**Script**

[scripts/01data_acquisition.py](scripts/01data_acquisition.py)

Holt **1‑minuten** Kerzen von **2020‑01‑01 bis 2025‑11‑21** und schreibt `symbol.parquet` Dateien zu `data/raw/Prices_1m_adj`.

- `columns`: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trade_count`,`vwap`, `symbol`   

Bar data AAPL Beispiel:

<img width="1462" height="906" alt="image" src="https://github.com/user-attachments/assets/84685933-70b6-4efa-bc9e-2b6509b03899" />

[scripts/fetch_nasdaq100_news_alpaca.py](scripts/fetch_nasdaq100_news_alpaca.py) 

Holt historische Nachrichtenartikel von **2020‑01‑01 bis 2025‑11‑21** und erstellt `symbol.parquet` Dateien zu `data/raw/News_alpaca`

- `colums`: `author`, `content`, `created_at`, `headline`, `id`, `images`, `source`, `summary`, `symbols`, `updated_at`, `url`,`symbol`

[scripts/news_sentiment_score.py](scripts/news_sentiment_score.py)  

Berechnet den Sentiment Score für jeden Nachrichtenartikel und erstellt`data/processed/nasdaq_news_with_sentiment.parquet` file.

- `columns`: `id`, `content`, `symbol`, `sentiment_score`,

Sentiment score AAPL Beispiel:

<img width="1266" height="697" alt="image" src="https://github.com/user-attachments/assets/a7f688a7-1021-47f4-aa78-4732f7ee627c" />

**API**

#### Alpaca Market Data API
Ruft historische 1-Minuten-Kerzendaten für NASDAQ-100 Symbole und QQQ über die Alpaca Data API ab. Die Daten werden im Parquet-Format unter `data/raw/Prices_1m_adj` gespeichert.


#### Parameter
- `symbol_or_symbols`: Liste von Tickern
- `timeframe`: 1Min (1-Minuten Bars)
- `start`: Start-Datum 2025-01-01
- `end`: End-Datum 2025-11-21
- `adjustment`: adjustiert für Splits & Dividenden


#### Alpaca Trading API
Filtern nach den Minuten, die innerhalb der regulären Handelszeiten des offiziellen US-Handelskalender liegen.


#### Parameter
- `start`: Start-Datum (01.01.2020)
- `end`: End-Datum 


#### Alpaca News API
Ruft Nachrichtenartikel seit 1.1.2020 für NASDAQ-bezogene Symbole über die Alpaca News API ab. Die Daten werden in `data/raw/News_alpaca` gespeichert.


#### Parameter 
- `symbols`: Symbole, die geladen werden müssen
- `start`: Start-Datum (01.01.2020)
- `end`: End-Datum (21.11.2025)
- `limit`: 50 (Anzahl Artikel pro Anfrage)
- `sort`: desc
- `include_content`: false (HTML-Content nicht nötig)


#### FinBERT API
Berechnet den sentiment score jedes Nachrichtenartikel. Die Daten werden in `data/processed/nasdaq_news_with_sentiment.parquet` gespeichert.


#### Parameter
- `text`: (Headline + Summary)
- `return_tensors`: pt (Ausgabe als PyTorch-Tensors)
- `truncation`: True (Text wird auf max_length abgeschnitten)
- `max_length`: 512 (maximale Tokenlänge)

