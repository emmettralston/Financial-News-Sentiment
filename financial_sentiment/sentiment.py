"""Sentiment analysis utilities using a FinBERT model."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


_FINBERT_MODEL = "ProsusAI/finbert"


def get_finbert_pipeline(device: int = -1):
    """Load a sentiment analysis pipeline using FinBERT.

    Parameters
    ----------
    device:
        Device to load the model on. ``-1`` uses CPU.
    """

    tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
    )


def analyze_headlines(headlines: Iterable[str], nlp_pipeline=None) -> pd.DataFrame:
    """Analyze a sequence of headlines and return sentiment scores.

    Parameters
    ----------
    headlines:
        Iterable of headline strings.
    nlp_pipeline:
        Preloaded HuggingFace ``pipeline``. If ``None``, a new FinBERT pipeline
        will be created. Passing a pipeline allows callers to cache the model
        across multiple invocations and simplifies testing.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``[headline, positive, negative, neutral, sentiment]``
        where ``sentiment`` is ``positive - negative``.
    """

    if nlp_pipeline is None:
        nlp_pipeline = get_finbert_pipeline()

    rows = []
    for text in headlines:
        scores = nlp_pipeline(text)[0]
        score_map = {s["label"].lower(): s["score"] for s in scores}
        rows.append(
            {
                "headline": text,
                **score_map,
                "sentiment": score_map.get("positive", 0) - score_map.get("negative", 0),
            }
        )

    return pd.DataFrame(rows)


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.Series:
    """Aggregate sentiment scores by calendar date."""

    if news_df.empty:
        return pd.Series(dtype=float)

    news_df = news_df.copy()
    news_df["date"] = news_df["datetime"].dt.date
    return news_df.groupby("date")["sentiment"].mean()
