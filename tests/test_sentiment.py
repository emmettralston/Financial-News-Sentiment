"""Unit tests for sentiment utilities."""

from __future__ import annotations

import pandas as pd

from financial_sentiment.sentiment import aggregate_daily_sentiment, analyze_headlines


class DummyPipeline:
    """Simple pipeline that returns deterministic scores for testing."""

    def __call__(self, text):  # type: ignore[override]
        return [
            [
                {"label": "positive", "score": 0.7},
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.1},
            ]
        ]


def test_analyze_headlines() -> None:
    df = analyze_headlines(["Stocks rally"], nlp_pipeline=DummyPipeline())
    assert "sentiment" in df.columns
    assert abs(df.loc[0, "sentiment"] - 0.5) < 1e-6


def test_aggregate_daily_sentiment() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2023-01-01 10:00", "2023-01-01 12:00", "2023-01-02 09:30"]
            ),
            "sentiment": [0.5, -0.1, 0.2],
        }
    )
    agg = aggregate_daily_sentiment(df)
    assert agg.loc[pd.Timestamp("2023-01-01").date()] == 0.2
    assert agg.loc[pd.Timestamp("2023-01-02").date()] == 0.2
