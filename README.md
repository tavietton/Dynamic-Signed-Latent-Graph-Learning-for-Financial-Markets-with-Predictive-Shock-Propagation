# Dynamic Signed Graph Stock Forecasting

This repository contains Python scripts for preprocessing equity OHLCV data, training a dynamic signed latent graph forecasting model, comparing it with baseline models, running statistical tests, and exporting graph-analysis outputs.

## Files

- `preprocess.py`: cleans raw OHLCV CSV files, aligns assets to a reference market calendar, constructs feature tensors, and saves `features.npy` and `meta.npz`.
- `train_dynamic_graph.py`: trains the dynamic signed latent graph model and saves test predictions, targets, metrics, and adjacency matrices.
- `compare_models_dynamic.py`: trains and evaluates baseline models against the dynamic model, then saves comparison results as CSV.
- `stat_test_dynamic_vs_lstm.py`: compares the dynamic model and LSTM baseline using paired forecasting-loss and accuracy tests.
- `analyze_dynamic_graph.py`: computes graph/node/edge summaries and produces diagnostic plots from the learned adjacency matrix.
- `dynamic_npy_to_csv.py`: converts the learned adjacency matrix and graph rankings from NumPy outputs to CSV/text files.

## Expected input data

Place raw OHLCV CSV files in a folder named `csv/`. The preprocessing script expects a reference calendar file named `csv/^GSPC.csv`.

The scripts assume CSV files contain standard OHLCV-style columns such as `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

## Recommended workflow

Run the scripts from the repository root in this order:

```bash
python preprocess.py
python train_dynamic_graph.py
python compare_models_dynamic.py
python stat_test_dynamic_vs_lstm.py
python analyze_dynamic_graph.py
python dynamic_npy_to_csv.py
```

## Main outputs

Typical generated outputs include:

- `features.npy`, `meta.npz`
- `dynamic_model.pt`
- `dynamic_training_log.csv`
- `dynamic_predictions_test.npy`, `dynamic_targets_test.npy`
- `dynamic_test_metrics.txt`
- `dynamic_adjacency_mean.npy`, `dynamic_adjacency_last.npy`
- `dynamic_forecasting_comparison.csv`
- `stat_test_dynamic_vs_lstm.csv`, `stat_test_dynamic_vs_lstm_summary.txt`
- `dynamic_node_metrics.csv`, `dynamic_top_edges_500.csv`, `dynamic_top_abs_edges_500.csv`
- graph diagnostic plots in PNG format

## Dependencies

The code uses:

- Python 3.10+
- NumPy
- pandas
- PyTorch
- matplotlib
- SciPy, optional but recommended for statistical testing

A minimal installation command is:

```bash
pip install numpy pandas torch matplotlib scipy
```

## Notes

- Model hyperparameters and data-processing settings are kept unchanged from the original experimental scripts.
- Data files, trained model checkpoints, and generated result files are not included in this archive.
- Random seeds are set inside the training scripts for reproducibility.
