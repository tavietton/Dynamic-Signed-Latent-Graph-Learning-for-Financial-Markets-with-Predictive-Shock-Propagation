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
META_FILE = "meta.npz"

WINDOW = 20
BATCH_SIZE = 4

EPOCHS = 100
LR = 1e-3
PATIENCE = 15

HIDDEN_DIM = 32
GRAPH_DIM = 16
MLP_DIM = 32
DROPOUT = 0.10

TOP_K = 20


LAMBDA_IC = 0.10
LAMBDA_DIR = 0.02
ALPHA_SPARSE = 0.01
BETA_VAR = 0.01
GAMMA_SMOOTH = 0.01

SEED = 42

OUTPUT_MODEL = "dynamic_model.pt"
OUTPUT_LOG = "dynamic_training_log.csv"
OUTPUT_ADJ_MEAN = "dynamic_adjacency_mean.npy"
OUTPUT_ADJ_LAST = "dynamic_adjacency_last.npy"
OUTPUT_PRED_TEST = "dynamic_predictions_test.npy"
OUTPUT_TARGET_TEST = "dynamic_targets_test.npy"
OUTPUT_METRICS = "dynamic_test_metrics.txt"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


class StockWindowDataset(Dataset):
    """
    features: numpy array with shape (T, N, d)

    For consistency with the previous code style:
    sample time t ranges from WINDOW to T-2.
    input  = features[t-WINDOW:t]      shape (WINDOW, N, d)
    target = features[t+1, :, 0]       shape (N,)
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

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


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


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    print("Dataset:")
    print("  features:", features.shape)
    print("  train samples:", len(train_ds))
    print("  val samples:", len(val_ds))
    print("  test samples:", len(test_ds))

    return train_loader, val_loader, test_loader


def cross_sectional_ic_torch(pred, target, eps=1e-8):
    """
    pred, target: shape (B, N)
    Returns mean cross-sectional Pearson correlation over batch.
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    cov = (pred_c * target_c).mean(dim=1)
    pred_std = torch.sqrt((pred_c ** 2).mean(dim=1) + eps)
    target_std = torch.sqrt((target_c ** 2).mean(dim=1) + eps)

    ic = cov / (pred_std * target_std + eps)
    return ic.mean()


def directional_soft_loss(pred, target, scale=10.0):
    """
    Soft directional objective.
    If pred and target have same sign, product tends to be positive.
    We minimize negative agreement.
    """
    return -torch.mean(torch.tanh(scale * pred) * torch.tanh(scale * target))


def compute_numpy_metrics(pred, target, eps=1e-8):
    """
    pred, target: numpy arrays with shape (num_samples, N)
    """
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))

    ics = []
    for i in range(pred.shape[0]):
        p = pred[i]
        y = target[i]

        p_c = p - p.mean()
        y_c = y - y.mean()

        denom = np.sqrt(np.mean(p_c ** 2) + eps) * np.sqrt(np.mean(y_c ** 2) + eps)
        ic = np.mean(p_c * y_c) / denom
        ics.append(ic)

    ic_mean = float(np.mean(ics))

    directional_accuracy = float(np.mean(np.sign(pred) == np.sign(target)))

    return {
        "mse": mse,
        "mae": mae,
        "ic": ic_mean,
        "directional_accuracy": directional_accuracy,
    }


class DynamicSignedGraphModel(nn.Module):
    """
    Dynamic Signed Latent Graph model.

    Architecture:
        Input:  (B, L, N, d)
        Shared LSTM per asset -> H: (B, N, hidden_dim)
        Dynamic graph from QK attention -> A_t: (B, N, N)
        Graph propagation -> graph representation
        Final prediction = temporal base prediction + graph residual
    """

    def __init__(
        self,
        N,
        d,
        hidden_dim=32,
        graph_dim=16,
        mlp_dim=32,
        top_k=20,
        dropout=0.10,
    ):
        super().__init__()

        self.N = N
        self.d = d
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.top_k = top_k

        self.asset_lstm = nn.LSTM(
            input_size=d,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.q_proj = nn.Linear(hidden_dim, graph_dim)
        self.k_proj = nn.Linear(hidden_dim, graph_dim)

        self.base_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

        self.graph_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def build_dynamic_adjacency(self, H):
        """
        H: shape (B, N, hidden_dim)

        Returns:
            A: shape (B, N, N)

        A[i, j] means source i -> target j.
        Propagation to target nodes uses A^T H, implemented as:
            Z_j = sum_i A[i, j] H_i
        """

        B, N, _ = H.shape

        Q = self.q_proj(H)
        K = self.k_proj(H)

        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.graph_dim)


        A = torch.tanh(scores)


        eye = torch.eye(N, device=H.device).unsqueeze(0)
        A = A * (1.0 - eye)


        if self.top_k is not None and self.top_k > 0 and self.top_k < N:
            abs_A = torch.abs(A)
            _, top_idx = torch.topk(abs_A, k=self.top_k, dim=2)

            mask = torch.zeros_like(A)
            mask.scatter_(2, top_idx, 1.0)

            A = A * mask


        valid_mask = (1.0 - eye)
        if self.top_k is not None and self.top_k > 0 and self.top_k < N:
            valid_mask = valid_mask * (A != 0.0).float()

        denom = valid_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean_A = (A * valid_mask).sum(dim=(1, 2), keepdim=True) / denom

        A = (A - mean_A) * valid_mask

        return A

    def forward(self, x, return_adjacency=False):
        """
        x: shape (B, L, N, d)
        """

        B, L, N, d = x.shape
        assert N == self.N
        assert d == self.d


        x_asset = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, d)

        _, (h_last, _) = self.asset_lstm(x_asset)
        h = h_last[-1]

        H = h.view(B, N, self.hidden_dim)

        A = self.build_dynamic_adjacency(H)


        Z = torch.matmul(A.transpose(1, 2), H)

        base_pred = self.base_head(H).squeeze(-1)
        graph_pred = self.graph_head(Z).squeeze(-1)

        pred = base_pred + self.residual_scale * graph_pred

        if return_adjacency:
            return pred, A

        return pred


def graph_regularization(A):
    """
    A: shape (B, N, N)
    """

    sparse_loss = torch.mean(torch.abs(A))
    var_loss = -torch.var(A)

    if A.shape[0] > 1:
        smooth_loss = torch.mean((A[1:] - A[:-1]) ** 2)
    else:
        smooth_loss = torch.tensor(0.0, device=A.device)

    return sparse_loss, var_loss, smooth_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_ic = 0.0

    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred, A = model(x, return_adjacency=True)

        mse_loss = torch.mean((pred - y) ** 2)
        ic = cross_sectional_ic_torch(pred, y)
        ic_loss = -ic
        dir_loss = directional_soft_loss(pred, y)

        sparse_loss, var_loss, smooth_loss = graph_regularization(A)

        loss = (
            mse_loss
            + LAMBDA_IC * ic_loss
            + LAMBDA_DIR * dir_loss
            + ALPHA_SPARSE * sparse_loss
            + BETA_VAR * var_loss
            + GAMMA_SMOOTH * smooth_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        total_ic += float(ic.item())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
        "ic": total_ic / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device, collect_adjacency=False):
    model.eval()

    preds = []
    targets = []
    adjs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if collect_adjacency:
            pred, A = model(x, return_adjacency=True)
            adjs.append(A.detach().cpu().numpy())
        else:
            pred = model(x, return_adjacency=False)

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    metrics = compute_numpy_metrics(preds, targets)

    if collect_adjacency:
        adjs = np.concatenate(adjs, axis=0)
        return metrics, preds, targets, adjs

    return metrics, preds, targets


def save_test_outputs(model, test_loader, device):
    metrics, preds, targets, adjs = evaluate(
        model,
        test_loader,
        device,
        collect_adjacency=True,
    )

    A_mean = adjs.mean(axis=0)
    A_last = adjs[-1]

    np.save(OUTPUT_ADJ_MEAN, A_mean)
    np.save(OUTPUT_ADJ_LAST, A_last)
    np.save(OUTPUT_PRED_TEST, preds)
    np.save(OUTPUT_TARGET_TEST, targets)

    with open(OUTPUT_METRICS, "w") as f:
        f.write("Dynamic Signed Graph Test Metrics\n")
        f.write("=================================\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8f}\n")

    print("\nSaved test outputs:")
    print(f"  {OUTPUT_ADJ_MEAN}")
    print(f"  {OUTPUT_ADJ_LAST}")
    print(f"  {OUTPUT_PRED_TEST}")
    print(f"  {OUTPUT_TARGET_TEST}")
    print(f"  {OUTPUT_METRICS}")

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8f}")

    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Cannot find {FEATURE_FILE}")

    features = np.load(FEATURE_FILE).astype(np.float32)

    T, N, d = features.shape
    print("Loaded features:", features.shape)

    train_loader, val_loader, test_loader = make_dataloaders(
        features=features,
        window=WINDOW,
        batch_size=BATCH_SIZE,
    )

    model = DynamicSignedGraphModel(
        N=N,
        d=d,
        hidden_dim=HIDDEN_DIM,
        graph_dim=GRAPH_DIM,
        mlp_dim=MLP_DIM,
        top_k=TOP_K,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_ic = -1e9
    best_state = None
    bad_epochs = 0

    logs = []

    print("\nStart training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device, collect_adjacency=False)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mse": train_stats["mse"],
            "train_ic": train_stats["ic"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_ic": val_metrics["ic"],
            "val_directional_accuracy": val_metrics["directional_accuracy"],
        }
        logs.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"train_mse={row['train_mse']:.6f} | "
            f"train_ic={row['train_ic']:.6f} | "
            f"val_mse={row['val_mse']:.6f} | "
            f"val_ic={row['val_ic']:.6f} | "
            f"val_dir={row['val_directional_accuracy']:.6f}"
        )


        if val_metrics["ic"] > best_val_ic:
            best_val_ic = val_metrics["ic"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "N": N,
                    "d": d,
                    "window": WINDOW,
                    "hidden_dim": HIDDEN_DIM,
                    "graph_dim": GRAPH_DIM,
                    "mlp_dim": MLP_DIM,
                    "top_k": TOP_K,
                    "dropout": DROPOUT,
                    "best_val_ic": best_val_ic,
                },
                OUTPUT_MODEL,
            )
        else:
            bad_epochs += 1

        pd.DataFrame(logs).to_csv(OUTPUT_LOG, index=False)

        if bad_epochs >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best val IC = {best_val_ic:.6f}")
            break

    print("\nTraining finished.")
    print(f"Best validation IC: {best_val_ic:.8f}")
    print(f"Saved model: {OUTPUT_MODEL}")
    print(f"Saved log: {OUTPUT_LOG}")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_test_outputs(model, test_loader, device)


if __name__ == "__main__":
    main()
