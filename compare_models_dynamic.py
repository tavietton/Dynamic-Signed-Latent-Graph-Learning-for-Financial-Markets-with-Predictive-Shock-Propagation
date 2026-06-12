import os
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


FEATURE_FILE = "features.npy"

DYNAMIC_PRED_FILE = "dynamic_predictions_test.npy"
DYNAMIC_TARGET_FILE = "dynamic_targets_test.npy"
DYNAMIC_METRICS_FILE = "dynamic_test_metrics.txt"

OUTPUT_CSV = "dynamic_forecasting_comparison.csv"


OUTPUT_LINEAR_PRED = "linear_predictions_test.npy"
OUTPUT_STATIC_GNN_PRED = "static_gnn_predictions_test.npy"
OUTPUT_PREVIOUS_GRAPH_PRED = "previous_signed_graph_predictions_test.npy"


OUTPUT_LSTM_PRED = "lstm_predictions_test.npy"
OUTPUT_LSTM_TARGET = "lstm_targets_test.npy"

WINDOW = 20


LSTM_BATCH_SIZE = 8
LSTM_EPOCHS = 50
LSTM_HIDDEN_DIM = 32
LSTM_LR = 1e-3
LSTM_PATIENCE = 10


BASELINE_BATCH_SIZE = 8
EPOCHS_LINEAR = 50
EPOCHS_STATIC_GNN = 50
EPOCHS_PREVIOUS_GRAPH = 100

LR = 1e-3
PATIENCE = 10

STATIC_GRAPH_TOPK = 20


ALPHA_SPARSE = 0.5
BETA_VAR = 10.0

SEED = 42


FORCE_RETRAIN_BASELINES = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


class StockWindowDataset(Dataset):
    """
    features: numpy array with shape (T, N, d)
    times: actual time indices t.
           input  = features[t-window:t]
           target = features[t+1, :, 0]
    """

    def __init__(self, features, times, window):
        self.features = features
        self.times = times
        self.window = window

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = int(self.times[idx])

        x = self.features[t - self.window:t]
        y = self.features[t + 1, :, 0]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def make_dataloaders(features, window, batch_size):
    T, N, d = features.shape


    times = np.arange(window, T - 1)
    num_samples = len(times)

    train_size = int(0.70 * num_samples)
    val_size = int(0.15 * num_samples)

    train_times = times[:train_size]
    val_times = times[train_size:train_size + val_size]
    test_times = times[train_size + val_size:]

    train_ds = StockWindowDataset(features, train_times, window)
    val_ds = StockWindowDataset(features, val_times, window)
    test_ds = StockWindowDataset(features, test_times, window)


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("Loaded features:", features.shape)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))

    return train_loader, val_loader, test_loader, train_times, val_times, test_times


def compute_numpy_metrics(pred, target, eps=1e-12):
    """
    pred, target: numpy arrays with shape (num_samples, N)
    """
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))

    ic_list = []

    for p, y in zip(pred, target):
        p_center = p - p.mean()
        y_center = y - y.mean()

        denom = np.sqrt(np.sum(p_center ** 2) * np.sum(y_center ** 2))

        if denom > eps:
            ic_list.append(float(np.sum(p_center * y_center) / denom))

    ic = float(np.mean(ic_list)) if len(ic_list) > 0 else np.nan
    directional_accuracy = float(np.mean(np.sign(pred) == np.sign(target)))

    return {
        "MSE": mse,
        "MAE": mae,
        "IC": ic,
        "Directional Accuracy": directional_accuracy,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, return_predictions=False):
    model.eval()

    preds = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)

        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    metrics = compute_numpy_metrics(preds, targets)

    if return_predictions:
        return metrics, preds, targets

    return metrics


class LinearBaseline(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):

        x_last = x[:, -1]
        out = self.fc(x_last).squeeze(-1)
        return out


class LSTMBaseline(nn.Module):
    def __init__(self, d, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):

        B, L, N, d = x.shape


        x_asset = x.permute(0, 2, 1, 3).contiguous()
        x_asset = x_asset.view(B * N, L, d)

        _, (h_last, _) = self.lstm(x_asset)
        h = h_last[-1]

        pred = self.head(h).view(B, N)

        return pred


def build_static_corr_graph(features, train_times, top_k=20):
    """
    Build a fixed signed correlation graph from training-period returns.

    A_static[i,j] is based on correlation between asset i and asset j.
    We use feature 0 as the return proxy.
    """


    target_times = train_times + 1
    target_times = target_times[target_times < features.shape[0]]

    R = features[target_times, :, 0]

    print("Building static correlation graph from returns:", R.shape)

    C = np.corrcoef(R.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    np.fill_diagonal(C, 0.0)

    N = C.shape[0]
    A = np.zeros_like(C, dtype=np.float32)

    for i in range(N):
        row = C[i]
        idx = np.argsort(-np.abs(row))[:top_k]
        A[i, idx] = row[idx]


    row_sum = np.sum(np.abs(A), axis=1, keepdims=True)
    A = A / (row_sum + 1e-8)

    return A.astype(np.float32)


class StaticGNNBaseline(nn.Module):
    def __init__(self, d, A_static):
        super().__init__()

        self.fc = nn.Linear(d, 1)

        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.register_buffer(
            "A_static",
            torch.tensor(A_static, dtype=torch.float32),
        )

    def forward(self, x):

        x_last = x[:, -1]

        r = self.fc(x_last).squeeze(-1)
        r = r - r.mean(dim=1, keepdim=True)

        graph_signal = torch.matmul(r, self.A_static)

        h = torch.stack([r, graph_signal], dim=-1)
        out = self.mlp(h).squeeze(-1)

        return out


class PreviousSignedLatentGraph(nn.Module):
    def __init__(self, N, d):
        super().__init__()

        self.fc = nn.Linear(d, 1)

        self.proj = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, N),
        )

        self.A_param = nn.Parameter(torch.randn(N, N) * 0.01)

    def forward(self, x):

        x_last = x[:, -1]

        r = self.fc(x_last).squeeze(-1)
        r = r - r.mean(dim=1, keepdim=True)

        A = torch.sigmoid(self.A_param)
        A = A - A.mean()

        out = torch.matmul(r, A)
        out = self.proj(out)

        return out, A


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    model_name,
    epochs=50,
    lr=1e-3,
    patience=10,
    use_graph_regularization=False,
):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    history = []

    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()

        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            if isinstance(out, tuple):
                pred, A = out
            else:
                pred = out
                A = None

            loss_pred = loss_fn(pred, y)

            if use_graph_regularization:
                if A is None:
                    raise ValueError("Graph regularization requested but model did not return adjacency A.")

                loss_sparse = torch.mean(torch.abs(A))
                loss_var = -torch.var(A)
                loss = loss_pred + ALPHA_SPARSE * loss_sparse + BETA_VAR * loss_var
            else:
                loss = loss_pred

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss_pred.item())

        train_mse = float(np.mean(train_losses))
        val_metrics = evaluate_model(model, val_loader, device)
        val_mse = val_metrics["MSE"]

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_metrics["MSE"],
            "val_mae": val_metrics["MAE"],
            "val_ic": val_metrics["IC"],
            "val_directional_accuracy": val_metrics["Directional Accuracy"],
        })

        print(
            f"{model_name} | Epoch {epoch:03d} | "
            f"TrainMSE={train_mse:.6f} | "
            f"ValMSE={val_metrics['MSE']:.6f} | "
            f"ValMAE={val_metrics['MAE']:.6f} | "
            f"ValIC={val_metrics['IC']:.6f} | "
            f"ValAcc={val_metrics['Directional Accuracy']:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping {model_name} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint saved for {model_name}.")

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_file = f"history_{model_name}.csv"
    history_df.to_csv(history_file, index=False)

    print(f"Best {model_name} ValMSE: {best_val_mse:.6f} at epoch {best_epoch}")
    print(f"Saved history: {history_file}")

    return model, history_df


def read_dynamic_metrics_from_txt(path):
    metrics = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            try:
                value = float(value)
            except Exception:
                continue

            if key == "mse":
                metrics["MSE"] = value
            elif key == "mae":
                metrics["MAE"] = value
            elif key == "ic":
                metrics["IC"] = value
            elif key == "directional_accuracy":
                metrics["Directional Accuracy"] = value

    return metrics


def get_dynamic_metrics():
    if os.path.exists(DYNAMIC_PRED_FILE) and os.path.exists(DYNAMIC_TARGET_FILE):
        pred = np.load(DYNAMIC_PRED_FILE)
        target = np.load(DYNAMIC_TARGET_FILE)
        return compute_numpy_metrics(pred, target)

    if os.path.exists(DYNAMIC_METRICS_FILE):
        return read_dynamic_metrics_from_txt(DYNAMIC_METRICS_FILE)

    raise FileNotFoundError(
        "Cannot find dynamic predictions/targets or dynamic_test_metrics.txt. "
        "Run train_dynamic_graph.py first."
    )


def main():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Cannot find {FEATURE_FILE}. Run preprocess.py first.")

    if not os.path.exists(DYNAMIC_PRED_FILE) or not os.path.exists(DYNAMIC_TARGET_FILE):
        raise FileNotFoundError(
            "Cannot find dynamic_predictions_test.npy or dynamic_targets_test.npy. "
            "Run train_dynamic_graph.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    features = np.load(FEATURE_FILE).astype(np.float32)
    T, N, d = features.shape

    print("Feature tensor:", features.shape)

    dynamic_metrics = get_dynamic_metrics()
    dynamic_pred = np.load(DYNAMIC_PRED_FILE)
    dynamic_target = np.load(DYNAMIC_TARGET_FILE)

    train_loader, val_loader, test_loader, train_times, val_times, test_times = make_dataloaders(
        features,
        WINDOW,
        BASELINE_BATCH_SIZE,
    )

    results = []


    results.append({
        "Model": "Dynamic Signed Latent Graph",
        **dynamic_metrics,
    })


    linear_model = LinearBaseline(d)

    linear_model, _ = train_model(
        linear_model,
        train_loader,
        val_loader,
        device,
        model_name="linear",
        epochs=EPOCHS_LINEAR,
        lr=LR,
        patience=PATIENCE,
        use_graph_regularization=False,
    )

    linear_metrics, linear_pred, y_true = evaluate_model(
        linear_model,
        test_loader,
        device,
        return_predictions=True,
    )

    np.save(OUTPUT_LINEAR_PRED, linear_pred)

    results.append({
        "Model": "Linear",
        **linear_metrics,
    })


    train_loader_lstm, val_loader_lstm, test_loader_lstm, _, _, _ = make_dataloaders(
        features,
        WINDOW,
        LSTM_BATCH_SIZE,
    )

    lstm_model = LSTMBaseline(d, hidden_dim=LSTM_HIDDEN_DIM)

    lstm_model, _ = train_model(
        lstm_model,
        train_loader_lstm,
        val_loader_lstm,
        device,
        model_name="lstm",
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
        use_graph_regularization=False,
    )

    lstm_metrics, lstm_pred, lstm_true = evaluate_model(
        lstm_model,
        test_loader_lstm,
        device,
        return_predictions=True,
    )

    np.save(OUTPUT_LSTM_PRED, lstm_pred)
    np.save(OUTPUT_LSTM_TARGET, lstm_true)

    results.append({
        "Model": "LSTM",
        **lstm_metrics,
    })


    A_static = build_static_corr_graph(
        features,
        train_times,
        top_k=STATIC_GRAPH_TOPK,
    )

    np.save("static_adjacency.npy", A_static)

    static_gnn_model = StaticGNNBaseline(d, A_static)

    static_gnn_model, _ = train_model(
        static_gnn_model,
        train_loader,
        val_loader,
        device,
        model_name="static_gnn",
        epochs=EPOCHS_STATIC_GNN,
        lr=LR,
        patience=PATIENCE,
        use_graph_regularization=False,
    )

    static_gnn_metrics, static_gnn_pred, _ = evaluate_model(
        static_gnn_model,
        test_loader,
        device,
        return_predictions=True,
    )

    np.save(OUTPUT_STATIC_GNN_PRED, static_gnn_pred)

    results.append({
        "Model": "Static-GNN",
        **static_gnn_metrics,
    })


    previous_graph_model = PreviousSignedLatentGraph(N, d)

    previous_graph_model, _ = train_model(
        previous_graph_model,
        train_loader,
        val_loader,
        device,
        model_name="previous_signed_graph",
        epochs=EPOCHS_PREVIOUS_GRAPH,
        lr=LR,
        patience=PATIENCE,
        use_graph_regularization=True,
    )

    previous_metrics, previous_pred, _ = evaluate_model(
        previous_graph_model,
        test_loader,
        device,
        return_predictions=True,
    )

    np.save(OUTPUT_PREVIOUS_GRAPH_PRED, previous_pred)

    results.append({
        "Model": "Previous Signed Latent Graph",
        **previous_metrics,
    })


    df = pd.DataFrame(results)


    df = df.sort_values(
        by=["IC", "MSE"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)


    print("\nForecasting comparison:")
    print(df.to_string(index=False))

    print("\nSaved:")
    print(" ", OUTPUT_CSV)
    print(" ", OUTPUT_LINEAR_PRED)
    print(" ", OUTPUT_LSTM_PRED)
    print(" ", OUTPUT_LSTM_TARGET)
    print(" ", OUTPUT_STATIC_GNN_PRED)
    print(" ", OUTPUT_PREVIOUS_GRAPH_PRED)
    print(" ", "static_adjacency.npy")

    best = df.iloc[0]
    print("\nBest model by IC:")
    print(f" {best['Model']}: {best['IC']:.8f}")


if __name__ == "__main__":
    main()
