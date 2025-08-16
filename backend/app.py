"""
This file will contian the flask backend code, this is where:
- set up the flask server
- define the routes
- call stock API to get stock data 
"""
import os
from copy import deepcopy
from flask import Flask, render_template, jsonify, request

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import your model pieces
from backend.models.lstm_model import (
    TimeSeriesDataset, LSTMModel, LSTMWithAttention
)
from backend.models.utils import (
    config, download_data, prepare_data_x, prepare_data_y, Normalizer,
    plot_series, plot_train_val, plot_predictions_zoom, plot_next_day_prediction
)

app = Flask(__name__, template_folder="templates", static_folder="static")

CHECKPOINT_DIR = "backend/models/checkpoints"
PLOT_DIR = "backend/static/plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html", summary=None, plots=None, next_day=None, ticker=None)


@app.route("/auto_complete")
def auto_complete():
    # Minimal placeholder — you can wire a real API later
    # Returns a couple of common tickers for demo
    query = (request.args.get("query") or "").upper()
    suggestions = [{"ticker": t, "name": n} for t, n in [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corp."),
        ("GOOGL", "Alphabet Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("TSLA", "Tesla Inc.")
    ] if t.startswith(query)]
    return jsonify(suggestions)


def train_or_load_for_symbol(symbol: str, epochs: int = 50):
    """Train once per symbol (cached), or load an existing checkpoint."""
    # 1) config override for symbol
    cfg = deepcopy(config)
    cfg["alpha_vantage"]["symbol"] = symbol

    # 2) download + preprocess
    dates, prices = download_data(cfg)
    scaler = Normalizer()
    x_norm = scaler.fit_transform(np.array(prices, dtype=np.float32))

    X, X_unseen = prepare_data_x(x_norm, cfg["data"]["window_size"])     # (N,T[,1]), (1,T[,1])
    y = prepare_data_y(x_norm, cfg["data"]["window_size"])               # (N,)

    split_index = int(X.shape[0] * cfg["data"]["train_split_size"])
    X_tr, y_tr = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    # 3) dataset + loader
    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds = TimeSeriesDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 4) model setup (auto-detect features)
    device = cfg["training"]["device"]
    sample_x, _ = next(iter(train_dl))
    input_size = sample_x.shape[-1] if sample_x.dim() >= 2 else 1

    use_attention = cfg["model"].get("use_attention", True)
    model = (LSTMWithAttention if use_attention else LSTMModel)(
        input_size=input_size,
        hidden_layer_size=cfg["model"]["lstm_size"],
        num_layers=cfg["model"]["num_lstm_layers"],
        output_size=1,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # 5) checkpoint paths per symbol
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{symbol}.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{symbol}_scaler.npy")

    if os.path.exists(ckpt_path) and os.path.exists(scaler_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        mu, sd = np.load(scaler_path)
        scaler.mu, scaler.sd = float(mu), float(sd)
    else:
        # train quick + cache
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

        def run_epoch(dataloader, train=True):
            model.train(train)
            loss_sum, count = 0.0, 0
            for xb, yb in dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                if xb.dim() == 2: xb = xb.unsqueeze(-1)  # [B,T] -> [B,T,1]
                if yb.dim() == 1: yb = yb.unsqueeze(1)   # [B] -> [B,1]
                if train:
                    optimizer.zero_grad()
                out = model(xb) if not use_attention else model(xb)[0]
                loss = criterion(out, yb)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                loss_sum += loss.item() * out.size(0)
                count += out.size(0)
            return loss_sum / max(count, 1)

        for epoch in range(epochs):
            tr = run_epoch(train_dl, True)
            vl = run_epoch(val_dl, False)
            if epoch % 5 == 0:
                print(f"[{symbol}] epoch {epoch+1}/{epochs} - train {tr:.6f}  val {vl:.6f}")

        torch.save(model.state_dict(), ckpt_path)
        np.save(scaler_path, np.array([scaler.mu, scaler.sd], dtype=np.float32))

    # 6) predictions
    def ensure3d(arr):
        t = torch.from_numpy(arr).float().to(device)
        if t.dim() == 2: t = t.unsqueeze(-1)
        elif t.dim() == 1: t = t.unsqueeze(0).unsqueeze(-1)
        return t

    with torch.no_grad():
        y_tr_pred = (model(ensure3d(X_tr))[0] if use_attention else model(ensure3d(X_tr))).cpu().numpy().reshape(-1)
        y_val_pred = (model(ensure3d(X_val))[0] if use_attention else model(ensure3d(X_val))).cpu().numpy().reshape(-1)
        next_norm = (model(ensure3d(X_unseen))[0] if use_attention else model(ensure3d(X_unseen))).cpu().numpy().reshape(-1)[0]

    y_tr_plot = scaler.inverse_transform(y_tr_pred)
    y_val_plot = scaler.inverse_transform(y_val_pred)
    next_price = float(scaler.inverse_transform(np.array([next_norm]))[0])

    # 7) build full-length overlays
    N = len(dates); win = cfg["data"]["window_size"]
    full_train = np.array([None] * N, dtype=object)
    full_val   = np.array([None] * N, dtype=object)
    split_idx_dates = int((N - win) * cfg["data"]["train_split_size"])
    full_train[win:win + len(y_tr_plot)] = y_tr_plot
    full_val[win + split_idx_dates: win + split_idx_dates + len(y_val_plot)] = y_val_plot

    # 8) write plots to static/plots/
    base = os.path.join(PLOT_DIR, symbol)
    os.makedirs(base, exist_ok=True)
    paths = {
        "series": os.path.join(base, f"{symbol}_series.png"),
        "trainval": os.path.join(base, f"{symbol}_trainval.png"),
        "zoom": os.path.join(base, f"{symbol}_zoom.png"),
        "next": os.path.join(base, f"{symbol}_nextday.png"),
    }
    # Save plots (no plt.show in web)
    plot_series(dates, prices, cfg, save_path=paths["series"], show=False)
    plot_train_val(dates, full_train, full_val, cfg, save_path=paths["trainval"], show=False)
    zoom = min(100, len(y_val_plot))
    plot_predictions_zoom(
        dates[-zoom:], np.array(prices[-zoom:]), np.array(y_val_plot[-zoom:]),
        cfg, save_path=paths["zoom"], show=False
    )
    plot_next_day_prediction(
        dates[-zoom:], np.array(prices[-zoom:]), np.array(y_val_plot[-zoom:]),
        next_price, cfg, save_path=paths["next"], show=False
    )

    # Return everything needed by template
    return {
        "dates": dates,
        "prices": prices,
        "next_price": next_price,
        "plot_urls": {
            # convert absolute paths to /static/... URLs
            k: "/" + p.split("backend/")[1] if "backend/" in p else "/static/" + p.split("static/")[1]
            for k, p in paths.items()
        }
    }


@app.route("/summary/<symbol>")
def summary(symbol):
    symbol = symbol.upper()
    try:
        result = train_or_load_for_symbol(symbol, epochs=50)  # you can tune epochs
        summary = {
            "name": f"{symbol} (fetched from API)",  # placeholder; wire a fundamentals API later
            "symbol": symbol,
            "sector": "—",
            "industry": "—",
            "marketCap": "—",
            "peRatio": "—",
        }
        return render_template(
            "index.html",
            summary=summary,
            plots=result["plot_urls"],
            next_day=result["next_price"],
            ticker=symbol
        )
    except Exception as e:
        # Render a simple error; you can improve this UX
        return render_template("index.html", summary=None, plots=None, next_day=None, ticker=symbol, error=str(e))


if __name__ == "__main__":
    # Basic dev run
    # export FLASK_APP=backend.app && flask run
    app.run(debug=True)



