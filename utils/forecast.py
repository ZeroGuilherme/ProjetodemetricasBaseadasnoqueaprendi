from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_daily_series(daily_series: pd.Series) -> pd.Series:
    series = daily_series.copy()
    if series.empty:
        return series.astype(float)

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()]

    series = series.sort_index().astype(float)
    normalized = series.groupby(series.index.normalize()).sum()
    full_index = pd.date_range(normalized.index.min(), normalized.index.max(), freq="D")
    return normalized.reindex(full_index, fill_value=0.0)


def _recursive_forecast(values: np.ndarray, horizon: int, window: int) -> np.ndarray:
    history = list(values.astype(float))
    forecast_values: list[float] = []

    for _ in range(horizon):
        effective_window = min(window, len(history))
        mean_value = float(np.mean(history[-effective_window:])) if effective_window > 0 else 0.0
        forecast_values.append(mean_value)
        history.append(mean_value)

    return np.array(forecast_values, dtype=float)


def forecast_moving_average(daily_series: pd.Series, horizon: int = 30, window: int = 7) -> dict[str, object]:
    """Forecast future revenue using a recursive moving-average baseline."""
    prepared_series = _prepare_daily_series(daily_series)
    if prepared_series.empty:
        return {
            "history": prepared_series,
            "forecast": pd.Series(dtype=float),
            "metrics": None,
        }

    effective_window = max(2, min(window, len(prepared_series))) if len(prepared_series) >= 2 else 1
    forecast_values = _recursive_forecast(prepared_series.values, horizon=horizon, window=effective_window)
    forecast_index = pd.date_range(prepared_series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_series = pd.Series(forecast_values, index=forecast_index, name="forecast")

    metrics = None
    test_size = min(14, max(7, len(prepared_series) // 5))
    if len(prepared_series) >= effective_window + test_size + 3:
        train = prepared_series.iloc[:-test_size]
        test = prepared_series.iloc[-test_size:]

        test_forecast_values = _recursive_forecast(train.values, horizon=test_size, window=effective_window)
        test_forecast = pd.Series(test_forecast_values, index=test.index)

        mae = float(np.mean(np.abs(test.values - test_forecast.values)))
        non_zero = test.values != 0
        if non_zero.any():
            mape = float(np.mean(np.abs((test.values[non_zero] - test_forecast.values[non_zero]) / test.values[non_zero])) * 100)
        else:
            mape = None

        metrics = {"mae": mae, "mape": mape, "test_size": int(test_size)}

    return {
        "history": prepared_series.rename("history"),
        "forecast": forecast_series,
        "metrics": metrics,
    }
