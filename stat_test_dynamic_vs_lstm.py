import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


FEATURE_FILE = "features.npy"

DYNAMIC_PRED_FILE = "dynamic_predictions_test.npy"
DYNAMIC_TARGET_FILE = "dynamic_targets_test.npy"

LSTM_PRED_FILE = "lstm_predictions_test.npy"
LSTM_TARGET_FILE = "lstm_targets_test.npy"

TRAIN_LSTM_IF_MISSING = True

WINDOW = 20
BATCH_SIZE = 8
LSTM_EPOCHS = 50
LR = 1e-3
PATIENCE = 10
HIDDEN_DIM = 32
SEED = 42

OUTPUT_CSV = "stat_test_dynamic_vs_lstm.csv"
OUTPUT_TXT = "stat_test_dynamic_vs_lstm_summary.txt"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


class StockWindowDataset(Dataset):
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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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
        x_asset = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, d)
        _, (h_last, _) = self.lstm(x_asset)
        h = h_last[-1]
        pred = self.head(h).view(B, N)
        return pred


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = np.mean((preds - targets) ** 2)
    return mse, preds, targets


def train_lstm_and_save_predictions():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Cannot find {FEATURE_FILE}")

    print("LSTM prediction file not found.")
    print("Training LSTM baseline to create lstm_predictions_test.npy ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    features = np.load(FEATURE_FILE).astype(np.float32)
    T, N, d = features.shape

    train_loader, val_loader, test_loader = make_dataloaders(
        features=features,
        window=WINDOW,
        batch_size=BATCH_SIZE,
    )

    model = LSTMBaseline(d=d, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        val_mse, _, _ = evaluate_model(model, val_loader, device)

        print(
            f"LSTM epoch {epoch:03d} | "
            f"train_mse={np.mean(train_losses):.6f} | "
            f"val_mse={val_mse:.6f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"Early stopping LSTM at epoch {epoch}. Best val MSE={best_val_mse:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, preds, targets = evaluate_model(model, test_loader, device)

    np.save(LSTM_PRED_FILE, preds)
    np.save(LSTM_TARGET_FILE, targets)

    print("Saved:")
    print(f"  {LSTM_PRED_FILE}")
    print(f"  {LSTM_TARGET_FILE}")
    print(f"LSTM test MSE from this run: {test_mse:.8f}")


def per_time_mse(pred, target):
    return np.mean((pred - target) ** 2, axis=1)


def per_time_mae(pred, target):
    return np.mean(np.abs(pred - target), axis=1)


def per_time_directional_error(pred, target):
    return np.mean(np.sign(pred) != np.sign(target), axis=1)


def per_time_directional_accuracy(pred, target):
    return np.mean(np.sign(pred) == np.sign(target), axis=1)


def per_time_ic(pred, target, eps=1e-12):
    ics = []
    for t in range(pred.shape[0]):
        p = pred[t]
        y = target[t]

        p_c = p - p.mean()
        y_c = y - y.mean()

        denom = np.sqrt(np.mean(p_c ** 2) + eps) * np.sqrt(np.mean(y_c ** 2) + eps)
        ics.append(np.mean(p_c * y_c) / denom)

    return np.array(ics)


def aggregate_metrics(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    ic = float(np.mean(per_time_ic(pred, target)))
    da = float(np.mean(np.sign(pred) == np.sign(target)))

    return {
        "MSE": mse,
        "MAE": mae,
        "IC": ic,
        "Directional Accuracy": da,
    }


def newey_west_long_run_variance(x, max_lag=None):
    """
    Newey-West long-run variance estimate for a mean-zero series.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)

    if max_lag is None:

        max_lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))

    max_lag = max(0, min(max_lag, T - 1))

    x = x - np.mean(x)

    gamma0 = np.mean(x * x)
    lrv = gamma0

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.mean(x[lag:] * x[:-lag])
        lrv += 2.0 * weight * gamma

    return lrv, max_lag


def diebold_mariano_test(loss_dynamic, loss_lstm, alternative="two-sided", max_lag=None):
    """
    loss_dynamic, loss_lstm: lower is better.
    d_t = loss_dynamic_t - loss_lstm_t.
    Negative mean difference favors Dynamic.

    alternative:
        "two-sided"
        "less"     tests Dynamic loss < LSTM loss
        "greater"  tests Dynamic loss > LSTM loss
    """
    loss_dynamic = np.asarray(loss_dynamic, dtype=float)
    loss_lstm = np.asarray(loss_lstm, dtype=float)

    if loss_dynamic.shape != loss_lstm.shape:
        raise ValueError("Loss series must have the same shape.")

    d = loss_dynamic - loss_lstm
    T = len(d)

    mean_d = np.mean(d)
    lrv, used_lag = newey_west_long_run_variance(d, max_lag=max_lag)

    if lrv <= 0:
        dm_stat = np.nan
        p_value = np.nan
    else:
        dm_stat = mean_d / np.sqrt(lrv / T)


        h = 1
        hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat = dm_stat * hln_factor

        if SCIPY_AVAILABLE:
            if alternative == "two-sided":
                p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=T - 1))
            elif alternative == "less":
                p_value = stats.t.cdf(dm_stat, df=T - 1)
            elif alternative == "greater":
                p_value = 1.0 - stats.t.cdf(dm_stat, df=T - 1)
            else:
                raise ValueError("Unknown alternative.")
        else:
            p_value = np.nan

    return {
        "mean_loss_dynamic": float(np.mean(loss_dynamic)),
        "mean_loss_lstm": float(np.mean(loss_lstm)),
        "mean_diff_dynamic_minus_lstm": float(mean_d),
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "nw_lag": int(used_lag),
        "n_obs": int(T),
    }


def paired_t_test(diff, alternative="two-sided"):
    """
    diff = metric_dynamic - metric_lstm.
    Positive diff favors Dynamic if metric is higher-better.
    """
    diff = np.asarray(diff, dtype=float)
    T = len(diff)
    mean_diff = np.mean(diff)
    sd = np.std(diff, ddof=1)

    if sd == 0:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = mean_diff / (sd / np.sqrt(T))

        if SCIPY_AVAILABLE:
            if alternative == "two-sided":
                p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=T - 1))
            elif alternative == "greater":
                p_value = 1.0 - stats.t.cdf(t_stat, df=T - 1)
            elif alternative == "less":
                p_value = stats.t.cdf(t_stat, df=T - 1)
            else:
                raise ValueError("Unknown alternative.")
        else:
            p_value = np.nan

    return {
        "mean_dynamic_minus_lstm": float(mean_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_obs": int(T),
    }


def paired_block_bootstrap(diff, block_size=10, n_boot=10000, seed=42):
    """
    Simple moving block bootstrap confidence interval for mean paired difference.
    Useful because adjacent test windows overlap and may be serially dependent.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    T = len(diff)

    n_blocks_needed = int(np.ceil(T / block_size))
    starts = np.arange(0, T - block_size + 1)

    boot_means = []

    for _ in range(n_boot):
        sampled = []
        for _ in range(n_blocks_needed):
            s = rng.choice(starts)
            sampled.append(diff[s:s + block_size])

        sampled = np.concatenate(sampled)[:T]
        boot_means.append(np.mean(sampled))

    boot_means = np.array(boot_means)

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])


    mean_diff = np.mean(diff)
    if mean_diff >= 0:
        p_two = 2.0 * np.mean(boot_means <= 0.0)
    else:
        p_two = 2.0 * np.mean(boot_means >= 0.0)

    p_two = min(float(p_two), 1.0)

    return {
        "mean_diff": float(mean_diff),
        "ci_2.5": float(ci_low),
        "ci_97.5": float(ci_high),
        "bootstrap_p_two_sided": p_two,
        "block_size": int(block_size),
        "n_boot": int(n_boot),
    }


def main():
    if not os.path.exists(DYNAMIC_PRED_FILE):
        raise FileNotFoundError(f"Cannot find {DYNAMIC_PRED_FILE}")

    if not os.path.exists(DYNAMIC_TARGET_FILE):
        raise FileNotFoundError(f"Cannot find {DYNAMIC_TARGET_FILE}")

    if not os.path.exists(LSTM_PRED_FILE):
        if TRAIN_LSTM_IF_MISSING:
            train_lstm_and_save_predictions()
        else:
            raise FileNotFoundError(
                f"Cannot find {LSTM_PRED_FILE}. "
                "You need LSTM predictions for a valid paired test."
            )

    pred_dynamic = np.load(DYNAMIC_PRED_FILE)
    target_dynamic = np.load(DYNAMIC_TARGET_FILE)

    pred_lstm = np.load(LSTM_PRED_FILE)

    if os.path.exists(LSTM_TARGET_FILE):
        target_lstm = np.load(LSTM_TARGET_FILE)
        if target_lstm.shape != target_dynamic.shape:
            raise ValueError("LSTM target shape differs from dynamic target shape.")
        if not np.allclose(target_lstm, target_dynamic, atol=1e-6):
            print("Warning: LSTM targets are not exactly equal to dynamic targets.")
            print("Using dynamic targets as common target.")

    target = target_dynamic

    if pred_dynamic.shape != pred_lstm.shape:
        raise ValueError(
            f"Prediction shapes differ: dynamic {pred_dynamic.shape}, LSTM {pred_lstm.shape}"
        )

    if pred_dynamic.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: pred {pred_dynamic.shape}, target {target.shape}"
        )

    print("Loaded:")
    print("  Dynamic predictions:", pred_dynamic.shape)
    print("  LSTM predictions:", pred_lstm.shape)
    print("  Targets:", target.shape)

    metrics_dynamic = aggregate_metrics(pred_dynamic, target)
    metrics_lstm = aggregate_metrics(pred_lstm, target)


    mse_dyn_t = per_time_mse(pred_dynamic, target)
    mse_lstm_t = per_time_mse(pred_lstm, target)

    mae_dyn_t = per_time_mae(pred_dynamic, target)
    mae_lstm_t = per_time_mae(pred_lstm, target)

    de_dyn_t = per_time_directional_error(pred_dynamic, target)
    de_lstm_t = per_time_directional_error(pred_lstm, target)

    ic_dyn_t = per_time_ic(pred_dynamic, target)
    ic_lstm_t = per_time_ic(pred_lstm, target)

    da_dyn_t = per_time_directional_accuracy(pred_dynamic, target)
    da_lstm_t = per_time_directional_accuracy(pred_lstm, target)


    dm_mse_two = diebold_mariano_test(mse_dyn_t, mse_lstm_t, alternative="two-sided")
    dm_mse_less = diebold_mariano_test(mse_dyn_t, mse_lstm_t, alternative="less")

    dm_mae_two = diebold_mariano_test(mae_dyn_t, mae_lstm_t, alternative="two-sided")
    dm_mae_less = diebold_mariano_test(mae_dyn_t, mae_lstm_t, alternative="less")

    dm_dir_two = diebold_mariano_test(de_dyn_t, de_lstm_t, alternative="two-sided")
    dm_dir_less = diebold_mariano_test(de_dyn_t, de_lstm_t, alternative="less")


    ic_diff = ic_dyn_t - ic_lstm_t
    da_diff = da_dyn_t - da_lstm_t

    ic_t_two = paired_t_test(ic_diff, alternative="two-sided")
    ic_t_greater = paired_t_test(ic_diff, alternative="greater")
    ic_boot = paired_block_bootstrap(ic_diff, block_size=10, n_boot=10000, seed=SEED)

    da_t_two = paired_t_test(da_diff, alternative="two-sided")
    da_t_greater = paired_t_test(da_diff, alternative="greater")
    da_boot = paired_block_bootstrap(da_diff, block_size=10, n_boot=10000, seed=SEED)

    rows = []

    def add_dm_rows(metric_name, two, one):
        rows.append({
            "test": "Diebold-Mariano",
            "metric": metric_name,
            "alternative": "two-sided",
            **two,
        })
        rows.append({
            "test": "Diebold-Mariano",
            "metric": metric_name,
            "alternative": "Dynamic lower loss than LSTM",
            **one,
        })

    add_dm_rows("MSE loss", dm_mse_two, dm_mse_less)
    add_dm_rows("MAE loss", dm_mae_two, dm_mae_less)
    add_dm_rows("Directional error loss", dm_dir_two, dm_dir_less)

    rows.append({
        "test": "Paired t-test",
        "metric": "IC",
        "alternative": "two-sided",
        "mean_dynamic_minus_lstm": ic_t_two["mean_dynamic_minus_lstm"],
        "t_stat": ic_t_two["t_stat"],
        "p_value": ic_t_two["p_value"],
        "n_obs": ic_t_two["n_obs"],
    })
    rows.append({
        "test": "Paired t-test",
        "metric": "IC",
        "alternative": "Dynamic IC greater than LSTM",
        "mean_dynamic_minus_lstm": ic_t_greater["mean_dynamic_minus_lstm"],
        "t_stat": ic_t_greater["t_stat"],
        "p_value": ic_t_greater["p_value"],
        "n_obs": ic_t_greater["n_obs"],
    })
    rows.append({
        "test": "Block bootstrap",
        "metric": "IC",
        "alternative": "two-sided",
        "mean_dynamic_minus_lstm": ic_boot["mean_diff"],
        "ci_2.5": ic_boot["ci_2.5"],
        "ci_97.5": ic_boot["ci_97.5"],
        "p_value": ic_boot["bootstrap_p_two_sided"],
        "block_size": ic_boot["block_size"],
        "n_boot": ic_boot["n_boot"],
    })

    rows.append({
        "test": "Paired t-test",
        "metric": "Directional accuracy",
        "alternative": "two-sided",
        "mean_dynamic_minus_lstm": da_t_two["mean_dynamic_minus_lstm"],
        "t_stat": da_t_two["t_stat"],
        "p_value": da_t_two["p_value"],
        "n_obs": da_t_two["n_obs"],
    })
    rows.append({
        "test": "Paired t-test",
        "metric": "Directional accuracy",
        "alternative": "Dynamic DA greater than LSTM",
        "mean_dynamic_minus_lstm": da_t_greater["mean_dynamic_minus_lstm"],
        "t_stat": da_t_greater["t_stat"],
        "p_value": da_t_greater["p_value"],
        "n_obs": da_t_greater["n_obs"],
    })
    rows.append({
        "test": "Block bootstrap",
        "metric": "Directional accuracy",
        "alternative": "two-sided",
        "mean_dynamic_minus_lstm": da_boot["mean_diff"],
        "ci_2.5": da_boot["ci_2.5"],
        "ci_97.5": da_boot["ci_97.5"],
        "p_value": da_boot["bootstrap_p_two_sided"],
        "block_size": da_boot["block_size"],
        "n_boot": da_boot["n_boot"],
    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_TXT, "w") as f:
        f.write("Statistical Tests: Dynamic Signed Latent Graph vs LSTM\n")
        f.write("=====================================================\n\n")

        f.write("Aggregate metrics\n")
        f.write("-----------------\n")
        f.write("Dynamic:\n")
        for k, v in metrics_dynamic.items():
            f.write(f"  {k}: {v:.8f}\n")
        f.write("LSTM:\n")
        for k, v in metrics_lstm.items():
            f.write(f"  {k}: {v:.8f}\n")

        f.write("\nImportant interpretation\n")
        f.write("------------------------\n")
        f.write("For DM tests, mean_diff_dynamic_minus_lstm < 0 favors Dynamic.\n")
        f.write("For IC/DA paired tests, mean_dynamic_minus_lstm > 0 favors Dynamic.\n")
        f.write("Use two-sided p-values for conservative reporting.\n")
        f.write("Use one-sided p-values only if the hypothesis Dynamic > LSTM was specified before testing.\n\n")

        f.write(results_df.to_string(index=False))
        f.write("\n")

    print("\nAggregate metrics:")
    print("Dynamic:", metrics_dynamic)
    print("LSTM:", metrics_lstm)

    print("\nStatistical test results:")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
