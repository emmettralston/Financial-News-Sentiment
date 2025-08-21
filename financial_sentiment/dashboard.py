"""Streamlit dashboard for visualizing sentiment-return correlations."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .data import compute_next_day_returns, fetch_headlines, fetch_price_history
from .sentiment import aggregate_daily_sentiment, analyze_headlines
from .model import run_regression


def main() -> None:
    """Render the Streamlit dashboard."""

    st.title("Financial News Sentiment vs. Stock Returns")

    ticker = st.text_input("Ticker symbol", value="AAPL")
    days = st.number_input("Days of history", min_value=1, max_value=365, value=30)

    end = date.today()
    start = end - timedelta(days=days)

    if st.button("Run analysis"):
        with st.spinner("Fetching data..."):
            headlines = fetch_headlines(ticker, start, end)
            prices = fetch_price_history(ticker, start, end + timedelta(days=1))

        if headlines.empty or prices.empty:
            st.warning("No data returned for selection.")
            return

        sentiments = analyze_headlines(headlines["headline"])
        headlines = pd.concat([headlines.reset_index(drop=True), sentiments], axis=1)
        daily_sentiment = aggregate_daily_sentiment(headlines)
        returns = compute_next_day_returns(prices)

        results = run_regression(daily_sentiment, returns)

        st.subheader("Regression Results")
        st.json(results)

        st.subheader("Daily Sentiment")
        st.line_chart(daily_sentiment)

        st.subheader("Next-day Returns")
        st.line_chart(returns)


if __name__ == "__main__":
    main()
