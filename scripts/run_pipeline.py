#!/usr/bin/env python
"""Command line interface for running the sentiment-return pipeline."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from financial_sentiment.data import (
    compute_next_day_returns,
    fetch_headlines,
    fetch_price_history,
)
from financial_sentiment.sentiment import aggregate_daily_sentiment, analyze_headlines
from financial_sentiment.model import run_regression


def main() -> None:
    """Entry point for the CLI."""

    parser = argparse.ArgumentParser(
        description="Correlate news sentiment with next-day stock returns."
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to analyze",
    )

    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    headlines = fetch_headlines(args.ticker, start, end)
    if headlines.empty:
        raise SystemExit("No headlines fetched")

    prices = fetch_price_history(args.ticker, start, end + timedelta(days=1))
    if prices.empty:
        raise SystemExit("No price data fetched")

    sentiments = analyze_headlines(headlines["headline"])
    headlines = headlines.reset_index(drop=True).join(sentiments)
    daily_sentiment = aggregate_daily_sentiment(headlines)
    returns = compute_next_day_returns(prices)

    results = run_regression(daily_sentiment, returns)
    print(results)


if __name__ == "__main__":
    main()
