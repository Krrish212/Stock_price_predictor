
import os
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None  # Will raise inside download_data if used without package


# -----------------------------
# Configuration (edit as needed)
# -----------------------------
config = {
    "alpha_vantage": {
        "key": os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_API_KEY"),
        "symbol": os.getenv("SYMBOL", "AAPL"),
        "outputsize": "full",
        "key_close": "4. close",
    },
    "data": {
        "window_size": 30,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
        "use_attention": True,
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "num_epoch": 70,
        "learning_rate": 0.001,
        "scheduler_step_size": 30,
    },
}


# -----------------------------
# Utilities
# -----------------------------

class Normalizer:
    """Simple standard-score normalizer with inverse transform."""
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        self.mu = float(np.mean(x))
        self.sd = float(np.std(x) + 1e-8)
        return (x - self.mu) / self.sd

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sd is None:
            raise ValueError("Normalizer not fitted. Call fit_transform first.")
        return (np.asarray(x, dtype=np.float32) * self.sd) + self.mu


def download_data(cfg=config) -> Tuple[List[str], List[float]]:
    """Download daily time series and return (dates_sorted, close_prices_sorted)."""
    if TimeSeries is None:
        raise ImportError("alpha_vantage not installed. Install with `pip install alpha_vantage`.")
    ts = TimeSeries(key=cfg["alpha_vantage"]["key"])
    data, _ = ts.get_daily(cfg["alpha_vantage"]["symbol"], outputsize=cfg["alpha_vantage"]["outputsize"])

    # Dates from oldest->newest
    dates = sorted(data.keys())
    closes = [float(data[d][cfg["alpha_vantage"]["key_close"]]) for d in dates]
    return dates, closes


def prepare_data_x(x: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_windows, last_unseen_window) using stride tricks; does not drop the final lookahead window."""
    x = np.asarray(x, dtype=np.float32)
    n_row = x.shape[0] - window_size + 1
    if n_row <= 0:
        raise ValueError(f"window_size ({window_size}) is too large for series length {x.shape[0]}")
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0])
    )
    return windows[:-1], windows[-1]


def prepare_data_y(x: np.ndarray, window_size: int) -> np.ndarray:
    """Use the next day value as the label; length = len(x) - window_size."""
    x = np.asarray(x, dtype=np.float32)
    return x[window_size:]


# -----------------------------
# Plotting helpers (define-only; call from training/inference code)
# -----------------------------

def plot_series(dates: List[str], prices: np.ndarray, cfg=config, title: str = "Daily close prices") -> None:
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, prices, label="Prices", color=cfg["plots"]["color_actual"])  # type: ignore
    xticks = [dates[i] if (i % cfg["plots"]["xticks_interval"] == 0) else None for i in range(len(dates))]  # type: ignore
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_train_val(dates: List[str],
                   y_train_plot: np.ndarray,
                   y_val_plot: np.ndarray,
                   cfg=config,
                   title: str = "Training vs validation") -> None:
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, y_train_plot, label="Prices (train)", color=cfg["plots"]["color_train"])  # type: ignore
    plt.plot(dates, y_val_plot, label="Prices (validation)", color=cfg["plots"]["color_val"])  # type: ignore
    xticks = [dates[i] if (i % cfg["plots"]["xticks_interval"] == 0) else None for i in range(len(dates))]  # type: ignore
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predictions_zoom(dates_subset: List[str],
                          actual_val: np.ndarray,
                          pred_val: np.ndarray,
                          cfg=config,
                          title: str = "Zoom in on predicted vs actual (validation)") -> None:
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates_subset, actual_val, label="Actual prices", color=cfg["plots"]["color_actual"])  # type: ignore
    plt.plot(dates_subset, pred_val, label="Predicted prices (validation)", color=cfg["plots"]["color_pred_val"])  # type: ignore
    plt.title(title)
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_next_day_prediction(plot_dates: List[str],
                             actual_val: np.ndarray,
                             pred_val: np.ndarray,
                             next_day_pred: float,
                             cfg=config,
                             title: str = "Predicting the next trading day close") -> None:
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_dates, actual_val, label="Actual prices", marker=".", markersize=10, color=cfg["plots"]["color_actual"])  # type: ignore
    plt.plot(plot_dates, pred_val,   label="Predicted prices (validation)", marker=".", markersize=10, color=cfg["plots"]["color_pred_val"])  # type: ignore
    plt.plot([plot_dates[-1]], [next_day_pred], label="Predicted price (next day)", marker=".", markersize=20, color=cfg["plots"]["color_pred_test"])  # type: ignore
    plt.title(title)
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


__all__ = [
    "config",
    "Normalizer",
    "download_data",
    "prepare_data_x",
    "prepare_data_y",
    "plot_series",
    "plot_train_val",
    "plot_predictions_zoom",
    "plot_next_day_prediction",
]
