import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .utils import (
    config,
    download_data,
    prepare_data_x,
    prepare_data_y,
    Normalizer,
    plot_series,
    plot_train_val,
    plot_predictions_zoom,
    plot_next_day_prediction
)

# ----------------------------
# Dataset
# ----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data_x, data_y):
        """
        data_x: np.ndarray, shape (N, T) or (N, T, 1)
        data_y: np.ndarray, shape (N,)
        """
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]  # (T,) or (T,1)
        y = self.data_y[idx]  # scalar

        # Ensure x is [T, 1]
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[:, None]
            x = torch.from_numpy(x).float()
        else:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            x = x.float()

        # y -> [1]
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y


# ----------------------------
# Attention module
# ----------------------------
class Attention(nn.Module):
    """
    Additive attention scorer: score_t = v^T tanh(W * h_t)
    For simplicity, we use a 1-layer linear scorer on h_t (commonly performs well).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)  # produces one score per timestep

    def forward(self, lstm_out):
        """
        lstm_out: [B, T, H]
        returns: context [B, H], weights [B, T, 1]
        """
        scores = self.scorer(lstm_out)              # [B, T, 1]
        weights = torch.softmax(scores, dim=1)      # softmax across time steps
        context = (lstm_out * weights).sum(dim=1)   # [B, H]
        return context, weights


# ----------------------------
# Models
# ----------------------------
class LSTMModel(nn.Module):
    """Vanilla LSTM -> Linear (no attention)"""
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [T, F] -> [1, T, F]
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))             # [B, T, H]
        out = self.linear(out[:, -1, :])            # last timestep -> [B, 1]
        return out


class LSTMWithAttention(nn.Module):
    """LSTM -> Attention over all timesteps -> Linear"""
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.attn = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))             # [B, T, H]
        context, weights = self.attn(out)           # context: [B, H]
        out = self.linear(context)                  # [B, 1]
        return out, weights                         # return weights for inspection


# ----------------------------
# Training/Eval
# ----------------------------
def run_epoch(model, dataloader, criterion, optimizer, device, is_training, use_attention: bool):
    model.train() if is_training else model.eval()
    running_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)   # [B, T, 1] or [B, T]
        y_batch = y_batch.to(device)   # [B, 1] or [B]

        # Shape guards
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)   # [B] -> [B, 1]

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            out = model(x_batch)
            if use_attention:
                outputs, _ = out           # model returns (pred, weights)
            else:
                outputs = out              # model returns pred only

            loss = criterion(outputs, y_batch)
            if is_training:
                loss.backward()
                # (optional) gradient clipping helps RNN stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        running_loss += loss.item() * outputs.size(0)

    return running_loss / len(dataloader.dataset)


# ----------------------------
# Main
# ----------------------------
def main():
    # === Load and prepare data ===
    data_date, data_close_price = download_data(config)
    plot_series(data_date, data_close_price, config)

    scaler = Normalizer()
    data_close_price_norm = scaler.fit_transform(np.array(data_close_price, dtype=np.float32))

    data_x, data_x_unseen = prepare_data_x(data_close_price_norm, config["data"]["window_size"])  # (N,T[,1]), (1,T[,1])
    data_y = prepare_data_y(data_close_price_norm, config["data"]["window_size"])                  # (N,)

    split_index = int(data_x.shape[0] * config["data"]["train_split_size"])
    data_x_train, data_y_train = data_x[:split_index], data_y[:split_index]
    data_x_val,   data_y_val   = data_x[split_index:], data_y[split_index:]

    train_dataset = TimeSeriesDataset(data_x_train, data_y_train)
    val_dataset   = TimeSeriesDataset(data_x_val,   data_y_val)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["training"]["batch_size"], shuffle=False)

    # === Model ===
    device = config["training"]["device"]
    # Auto-detect feature dimension from a sample batch
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1] if sample_x.dim() >= 2 else 1

    use_attention = config["model"].get("use_attention", True)
    if use_attention:
        model = LSTMWithAttention(
            input_size=input_size,
            hidden_layer_size=config["model"]["lstm_size"],
            num_layers=config["model"]["num_lstm_layers"],
            output_size=1,
            dropout=config["model"]["dropout"]
        ).to(device)
    else:
        model = LSTMModel(
            input_size=input_size,
            hidden_layer_size=config["model"]["lstm_size"],
            num_layers=config["model"]["num_lstm_layers"],
            output_size=1,
            dropout=config["model"]["dropout"]
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # === Training loop ===
    for epoch in range(config["training"]["num_epoch"]):
        loss_train = run_epoch(model, train_loader, criterion, optimizer, device, True,  use_attention)
        loss_val   = run_epoch(model, val_loader,   criterion, optimizer, device, False, use_attention)
        print(f"Epoch {epoch+1}: train loss={loss_train:.6f}, val loss={loss_val:.6f}")

    # === Predictions ===
    with torch.no_grad():
        def _ensure_3d(arr):
            t = torch.from_numpy(arr).float().to(device)
            if t.dim() == 2: t = t.unsqueeze(-1)              # [B,T] -> [B,T,1]
            elif t.dim() == 1: t = t.unsqueeze(0).unsqueeze(-1)  # [T] -> [1,T,1]
            return t

        x_tr   = _ensure_3d(data_x_train)
        x_val  = _ensure_3d(data_x_val)
        x_next = _ensure_3d(data_x_unseen)

        if use_attention:
            y_train_pred, _ = model(x_tr); y_train_pred = y_train_pred.cpu().numpy().reshape(-1)
            y_val_pred,   _ = model(x_val); y_val_pred   = y_val_pred.cpu().numpy().reshape(-1)
            next_day_pred_norm, _ = model(x_next); next_day_pred_norm = next_day_pred_norm.cpu().numpy().reshape(-1)[0]
        else:
            y_train_pred = model(x_tr).cpu().numpy().reshape(-1)
            y_val_pred   = model(x_val).cpu().numpy().reshape(-1)
            next_day_pred_norm = model(x_next).cpu().numpy().reshape(-1)[0]

    # Inverse-transform back to price space
    y_train_plot = scaler.inverse_transform(y_train_pred)
    y_val_plot   = scaler.inverse_transform(y_val_pred)
    next_day_pred = float(scaler.inverse_transform(np.array([next_day_pred_norm]))[0])

    # === Plot results ===
    # Align train/val overlays to the full date timeline
    N = len(data_date)
    win = config["data"]["window_size"]
    full_train = np.array([None] * N, dtype=object)
    full_val   = np.array([None] * N, dtype=object)

    split_idx_dates = int((N - win) * config["data"]["train_split_size"])
    full_train[win:win + len(y_train_plot)] = y_train_plot
    full_val[win + split_idx_dates: win + split_idx_dates + len(y_val_plot)] = y_val_plot

    plot_train_val(data_date, full_train, full_val, config)

    # Zoom + next day
    zoom_size = min(100, len(y_val_plot))
    plot_predictions_zoom(
        data_date[-zoom_size:],
        np.array([p for p in data_close_price][-zoom_size:]),
        np.array(y_val_plot[-zoom_size:]),
        config
    )
    plot_next_day_prediction(
        data_date[-zoom_size:],
        np.array([p for p in data_close_price][-zoom_size:]),
        np.array(y_val_plot[-zoom_size:]),
        next_day_pred,
        config
    )

    # Save model
    torch.save(model.state_dict(), "lstm_model.pt")
    print("Model saved as lstm_model.pt")


if __name__ == "__main__":
    main()