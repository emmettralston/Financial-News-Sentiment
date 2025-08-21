# Financial News Sentiment

This project correlates sentiment extracted from financial news headlines with
subsequent stock price movements. It fetches headlines from Yahoo Finance,
computes sentiment using a FinBERT transformer model and performs a regression
against next-day stock returns. A Streamlit dashboard provides an interactive
view of the results.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command line pipeline

Run the end-to-end analysis for a ticker:

```bash
python scripts/run_pipeline.py AAPL --days 30
```

### Streamlit dashboard

Launch the dashboard:

```bash
streamlit run financial_sentiment/dashboard.py
```

## Tests

```bash
pytest
```
