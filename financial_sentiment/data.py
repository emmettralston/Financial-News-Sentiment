"""Data retrieval utilities for financial news and stock prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf


@dataclass
class NewsItem:
    """Representation of a single news headline."""

    ticker: str
    datetime: datetime
    headline: str
    publisher: str


def fetch_headlines(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch news headlines for a ticker between *start* and *end* dates.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    start:
        Inclusive start datetime.
    end:
        Inclusive end datetime.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``[ticker, datetime, headline, publisher]``.
    """

    ticker_obj = yf.Ticker(ticker)
    news_items = getattr(ticker_obj, "news", [])

    rows = []
    for item in news_items:
        publish_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))
        if start <= publish_time <= end:
            rows.append(
                NewsItem(
                    ticker=ticker,
                    datetime=publish_time,
                    headline=item.get("title", ""),
                    publisher=item.get("publisher", ""),
                ).__dict__
            )

    return pd.DataFrame(rows)


def fetch_price_history(ticker: str, start: datetime, end: datetime) -> pd.Series:
    """Download daily adjusted close prices for *ticker* in the date range."""

    data = yf.download(ticker, start=start, end=end, progress=False)
    return data["Adj Close"].sort_index()


def compute_next_day_returns(prices: pd.Series) -> pd.Series:
    """Compute next-day returns from a series of prices.

    The return on day *t* is defined as ``(price[t+1] - price[t]) / price[t]``.
    The final day is dropped as it has no subsequent value.
    """

    returns = prices.pct_change().shift(-1)
    return returns.dropna()
