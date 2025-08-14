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

        # Ensure x is 2-D: [T, 1]
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[:, None]
            x = torch.from_numpy(x).float()
        else:
            # tensor fallback
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            x = x.float()

        # Ensure y is [1]
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y


# ----------------------------
# Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Expect x: [B, T, F]
        # If a stray [T, F] sneaks in, add batch dim
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_layer_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))   # [B, T, H]
        out = self.linear(out[:, -1, :])  # [B, 1]
        return out


# ----------------------------
# Training/Eval
# ----------------------------
def run_epoch(model, dataloader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()
    running_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)  # [B, T, 1] or [T, 1] or [B, T]
        y_batch = y_batch.to(device)  # [B, 1] or [B]

        # ---- Shape guards ----
        # Ensure feature dim exists: [B, T] -> [B, T, 1]
        if x_batch.dim() == 2:
            # If this is [T, 1] with no batch, add batch in forward.
            # But typical offender here is [B, T] (missing feature).
            # Add feature dim at the end.
            x_batch = x_batch.unsqueeze(-1)  # -> [B, T, 1] OR [T, 1, 1] (handled in forward)
        # Ensure targets are [B, 1]
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
        # ----------------------

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(x_batch)          # [B, 1]
            loss = criterion(outputs, y_batch)
            if is_training:
                loss.backward()
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

    data_x, data_x_unseen = prepare_data_x(data_close_price_norm, config["data"]["window_size"])  # X: (N, T) or (N, T, 1), X_unseen: (1, T) or (1, T, 1)
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
    model = LSTMModel(
        input_size=config["model"]["input_size"],    # 1
        hidden_layer_size=config["model"]["lstm_size"],
        num_layers=config["model"]["num_lstm_layers"],
        output_size=1,
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # === Training loop ===
    for epoch in range(config["training"]["num_epoch"]):
        loss_train = run_epoch(model, train_loader, criterion, optimizer, device, True)
        loss_val   = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        print(f"Epoch {epoch+1}: train loss={loss_train:.6f}, val loss={loss_val:.6f}")

    # === Predictions ===
    with torch.no_grad():
        # Ensure [B, T, 1] before calling model
        def _ensure_3d(arr):
            t = torch.from_numpy(arr).float().to(device)
            if t.dim() == 2:              # [B, T]
                t = t.unsqueeze(-1)       # -> [B, T, 1]
            elif t.dim() == 1:            # [T]
                t = t.unsqueeze(0).unsqueeze(-1)  # -> [1, T, 1]
            return t

        x_tr   = _ensure_3d(data_x_train)
        x_val  = _ensure_3d(data_x_val)
        x_next = _ensure_3d(data_x_unseen)

        y_train_pred = model(x_tr).cpu().numpy().reshape(-1)      # [B]
        y_val_pred   = model(x_val).cpu().numpy().reshape(-1)     # [B]
        next_day_pred_norm = model(x_next).cpu().numpy().reshape(-1)[0]  # scalar

    # Inverse-transform back to price space
    y_train_plot = scaler.inverse_transform(y_train_pred)
    y_val_plot   = scaler.inverse_transform(y_val_pred)
    next_day_pred = float(scaler.inverse_transform(np.array([next_day_pred_norm]))[0])

    # === Plot results ===
    # Plot train vs val over the full timeline: align arrays to the full date list
    N = len(data_date)
    win = config["data"]["window_size"]
    full_train = np.array([None] * N, dtype=object)
    full_val   = np.array([None] * N, dtype=object)

    # Place them correctly in the timeline
    split_index = int((N - win) * config["data"]["train_split_size"])
    # training predictions span indices [win : win+len(y_train_plot))
    full_train[win:win + len(y_train_plot)] = y_train_plot
    # validation predictions span [win+split_offset : ]
    full_val[win + split_index: win + split_index + len(y_val_plot)] = y_val_plot

    plot_train_val(data_date, full_train, full_val, config)

    # Zoomed-in on recent validation window
    zoom_size = min(100, len(y_val_plot))
    plot_predictions_zoom(
        data_date[-zoom_size:],
        np.array(data_close_price[-zoom_size:]),
        np.array(y_val_plot[-zoom_size:]),
        config
    )

    # Next-day prediction plot (uses same recent window)
    plot_next_day_prediction(
        data_date[-zoom_size:],
        np.array(data_close_price[-zoom_size:]),
        np.array(y_val_plot[-zoom_size:]),
        next_day_pred,
        config
    )

    # === Save model ===
    torch.save(model.state_dict(), "lstm_model.pt")
    print("Model saved as lstm_model.pt")


if __name__ == "__main__":
    main()