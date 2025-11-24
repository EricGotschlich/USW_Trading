"""
Berechnet Sentiment-Scores für alle News-Parquet-Dateien in data/raw/News_raw
und speichert das Ergebnis in data/processed/nasdaq_news_with_sentiment.parquet.

Sentiment-Modell: ProsusAI/finbert
Score: p(positiv) - p(negativ), Wert in [-1, +1]
"""

import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_NAME = "ProsusAI/finbert"


# ------------------------ Modell laden ------------------------ #

def load_finbert():
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return tokenizer, model, device


# ------------------------ Text-Bereinigung ------------------------ #

def clean_text(val) -> str:
    """
    Versucht, beliebige Werte (bytes, Listen von ints, etc.) in einen
    sinnvollen String umzuwandeln.
    """
    if val is None:
        return ""

    # Schon String?
    if isinstance(val, str):
        return val.strip()

    # Bytes -> UTF-8
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # Liste/Tuple von ints (z.B. [65,65,80,76] -> "AAPL")
    if isinstance(val, (list, tuple)) and val and isinstance(val[0], int):
        try:
            return bytes(val).decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # Fallback: einfach str()
    return str(val).strip()


def build_text_for_row(row: pd.Series) -> str:
    """
    Baut einen Text für Sentiment-Analyse aus typischen Spalten der News:
    title, summary, content, headline.
    """
    parts = []
    for col in ["title", "summary", "content", "headline"]:
        if col in row.index:
            txt = clean_text(row[col])
            if txt:
                parts.append(txt)

    return ". ".join(parts)


# ------------------------ Sentiment für einen Text ------------------------ #

def get_sentiment(text, tokenizer, model, device) -> float:
    """
    Berechnet Sentiment-Score in [-1, +1] für einen Text.
    """
    text = clean_text(text)
    if not text:
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT: [negativ, neutral, positiv]
    probs = probs.cpu().numpy()[0]
    score = probs[2] - probs[0]
    return float(score)


# ------------------------ Hauptfunktion ------------------------ #

def main():
    project_root = Path(__file__).resolve().parents[1]
    news_dir = project_root / "data" / "raw" / "News_raw"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    news_files = sorted(news_dir.glob("*_news.parquet"))
    if not news_files:
        raise FileNotFoundError(f"Keine *_news.parquet in {news_dir} gefunden")

    print(f"Found {len(news_files)} news parquet files")

    dfs = []
    for f in news_files:
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f"  {f.name}: {len(df)} rows")

    news = pd.concat(dfs, ignore_index=True)
    print(f"Total news items: {len(news)}")

    # Debug: Spalten einmal anzeigen
    print("Columns in news DataFrame:")
    print(list(news.columns))

    # Text pro News bauen
    news["text_for_sentiment"] = news.apply(build_text_for_row, axis=1)

    # FinBERT laden
    tokenizer, model, device = load_finbert()

    # Sentiment berechnen
    scores = []
    for text in tqdm(news["text_for_sentiment"], desc="Calculating sentiment"):
        scores.append(get_sentiment(text, tokenizer, model, device))

    news["sentiment_score"] = scores
    news.drop(columns=["text_for_sentiment"], inplace=True)

    out_path = out_dir / "nasdaq_news_with_sentiment.parquet"
    news.to_parquet(out_path, index=False)

    print(f"\nSentiment analysis complete, saved to: {out_path}")
    print("Sentiment stats:")
    print(news["sentiment_score"].describe())


if __name__ == "__main__":
    main()
