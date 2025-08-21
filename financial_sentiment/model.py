"""Model utilities for correlating sentiment with stock returns."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_regression_data(
    sentiment: pd.Series, returns: pd.Series
) -> pd.DataFrame:
    """Join sentiment and return series on date and drop missing values."""

    df = pd.concat([sentiment.rename("sentiment"), returns.rename("return")], axis=1)
    return df.dropna()


def run_regression(sentiment: pd.Series, returns: pd.Series) -> Dict[str, float]:
    """Fit a linear regression of returns on sentiment.

    Parameters
    ----------
    sentiment:
        Series of daily sentiment scores indexed by date.
    returns:
        Series of next-day returns indexed by date.

    Returns
    -------
    dict
        Dictionary with ``coef``, ``intercept`` and ``r2`` values.
    """

    df = prepare_regression_data(sentiment, returns)
    if df.empty:
        return {"coef": 0.0, "intercept": 0.0, "r2": 0.0}

    X = df[["sentiment"]]
    y = df["return"]

    model = LinearRegression()
    model.fit(X, y)

    return {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
    }
