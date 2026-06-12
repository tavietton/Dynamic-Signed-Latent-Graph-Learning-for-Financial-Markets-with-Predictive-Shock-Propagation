import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ADJ_FILE = "dynamic_adjacency_mean.npy"
META_FILE = "meta.npz"

TOP_EDGES = 500

OUTPUT_SUMMARY = "dynamic_graph_summary.txt"
OUTPUT_NODE_METRICS = "dynamic_node_metrics.csv"
OUTPUT_TOP_EDGES = "dynamic_top_edges_500.csv"
OUTPUT_TOP_ABS_EDGES = "dynamic_top_abs_edges_500.csv"

OUTPUT_HEATMAP = "dynamic_adjacency_heatmap.png"
OUTPUT_DEGREE_DIST = "dynamic_degree_dist.png"
OUTPUT_ABS_STRENGTH_DIST = "dynamic_abs_out_strength_dist.png"
OUTPUT_POS_NEG_STRENGTH = "dynamic_positive_vs_negative_strength.png"


def load_data():
    if not os.path.exists(ADJ_FILE):
        raise FileNotFoundError(f"Cannot find {ADJ_FILE}")

    A = np.load(ADJ_FILE)

    if A.ndim != 2:
        raise ValueError(f"Adjacency matrix must be 2D, got shape {A.shape}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}")

    N = A.shape[0]

    if os.path.exists(META_FILE):
        meta = np.load(META_FILE, allow_pickle=True)
        if "stock_names" in meta.files:
            stock_names = meta["stock_names"]
        else:
            stock_names = np.array([f"Stock_{i}" for i in range(N)])
    else:
        stock_names = np.array([f"Stock_{i}" for i in range(N)])

    stock_names = np.array(stock_names).astype(str)

    if len(stock_names) != N:
        raise ValueError(
            f"Number of stock names ({len(stock_names)}) does not match adjacency size ({N})."
        )

    print("Loaded adjacency:", A.shape)
    print("Loaded stock names:", len(stock_names))

    return A, stock_names


def compute_node_metrics(A, stock_names):
    signed_degree = A.sum(axis=1)
    abs_out_strength = np.abs(A).sum(axis=1)
    positive_out_strength = np.maximum(A, 0.0).sum(axis=1)
    negative_out_strength = np.abs(np.minimum(A, 0.0)).sum(axis=1)

    metrics = pd.DataFrame({
        "stock": stock_names,
        "signed_degree": signed_degree,
        "abs_out_strength": abs_out_strength,
        "positive_out_strength": positive_out_strength,
        "negative_out_strength": negative_out_strength,
    })

    return metrics


def compute_edge_metrics(A, stock_names, top_edges=500):
    edges = []
    N = A.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            weight = float(A[i, j])

            edges.append({
                "source": stock_names[i],
                "target": stock_names[j],
                "weight": weight,
                "abs_weight": abs(weight),
            })

    edges_df = pd.DataFrame(edges)

    top_positive_edges = (
        edges_df
        .sort_values("weight", ascending=False)
        .head(top_edges)
        .reset_index(drop=True)
    )

    top_abs_edges = (
        edges_df
        .sort_values("abs_weight", ascending=False)
        .head(top_edges)
        .reset_index(drop=True)
    )

    return edges_df, top_positive_edges, top_abs_edges


def make_plots(A, metrics):
    n_show = min(50, A.shape[0])

    plt.figure(figsize=(8, 6))
    plt.imshow(A[:n_show, :n_show], aspect="auto")
    plt.colorbar()
    plt.title("Dynamic Adjacency Heatmap (first 50 stocks)")
    plt.xlabel("Target asset")
    plt.ylabel("Source asset")
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(metrics["signed_degree"], bins=40)
    plt.title("Signed Degree Distribution")
    plt.xlabel("Signed degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DEGREE_DIST, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(metrics["abs_out_strength"], bins=40)
    plt.title("Absolute Out-Strength Distribution")
    plt.xlabel("Absolute out-strength")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_ABS_STRENGTH_DIST, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        metrics["positive_out_strength"],
        metrics["negative_out_strength"],
        s=18,
        alpha=0.70,
    )
    plt.title("Positive vs Negative Out-Strength")
    plt.xlabel("Positive out-strength")
    plt.ylabel("Negative out-strength")
    plt.tight_layout()
    plt.savefig(OUTPUT_POS_NEG_STRENGTH, dpi=150)
    plt.close()

    print("Saved plots:")
    print(f"  {OUTPUT_HEATMAP}")
    print(f"  {OUTPUT_DEGREE_DIST}")
    print(f"  {OUTPUT_ABS_STRENGTH_DIST}")
    print(f"  {OUTPUT_POS_NEG_STRENGTH}")


def save_graph_summary(A, metrics, top_positive_edges):
    signed_degree = metrics["signed_degree"].values
    abs_out_strength = metrics["abs_out_strength"].values
    positive_out_strength = metrics["positive_out_strength"].values
    negative_out_strength = metrics["negative_out_strength"].values

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Dynamic Graph Summary\n")
        f.write("=====================\n")
        f.write(f"Adjacency file: {ADJ_FILE}\n")
        f.write(f"Adjacency shape: {A.shape}\n")
        f.write(f"Number of stocks: {A.shape[0]}\n")
        f.write("\n")

        f.write("Edge weights\n")
        f.write("------------\n")
        f.write(f"Min weight: {A.min():.10f}\n")
        f.write(f"Max weight: {A.max():.10f}\n")
        f.write(f"Mean weight: {A.mean():.10f}\n")
        f.write(f"Std weight: {A.std():.10f}\n")
        f.write("\n")

        f.write("Signed degree\n")
        f.write("-------------\n")
        f.write(f"Min signed degree: {signed_degree.min():.10f}\n")
        f.write(f"Max signed degree: {signed_degree.max():.10f}\n")
        f.write(f"Mean signed degree: {signed_degree.mean():.10f}\n")
        f.write(f"Std signed degree: {signed_degree.std():.10f}\n")
        f.write("\n")

        f.write("Absolute out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min abs out-strength: {abs_out_strength.min():.10f}\n")
        f.write(f"Max abs out-strength: {abs_out_strength.max():.10f}\n")
        f.write(f"Mean abs out-strength: {abs_out_strength.mean():.10f}\n")
        f.write(f"Std abs out-strength: {abs_out_strength.std():.10f}\n")
        f.write("\n")

        f.write("Positive out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min positive out-strength: {positive_out_strength.min():.10f}\n")
        f.write(f"Max positive out-strength: {positive_out_strength.max():.10f}\n")
        f.write(f"Mean positive out-strength: {positive_out_strength.mean():.10f}\n")
        f.write(f"Std positive out-strength: {positive_out_strength.std():.10f}\n")
        f.write("\n")

        f.write("Negative out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min negative out-strength: {negative_out_strength.min():.10f}\n")
        f.write(f"Max negative out-strength: {negative_out_strength.max():.10f}\n")
        f.write(f"Mean negative out-strength: {negative_out_strength.mean():.10f}\n")
        f.write(f"Std negative out-strength: {negative_out_strength.std():.10f}\n")
        f.write("\n")

        f.write("Top 10 by signed degree\n")
        f.write("-----------------------\n")
        top_signed = metrics.sort_values("signed_degree", ascending=False).head(10)
        for _, row in top_signed.iterrows():
            f.write(f"{row['stock']}: {row['signed_degree']:.10f}\n")
        f.write("\n")

        f.write("Top 10 by absolute out-strength\n")
        f.write("--------------------------------\n")
        top_abs = metrics.sort_values("abs_out_strength", ascending=False).head(10)
        for _, row in top_abs.iterrows():
            f.write(
                f"{row['stock']}: "
                f"abs={row['abs_out_strength']:.10f}, "
                f"pos={row['positive_out_strength']:.10f}, "
                f"neg={row['negative_out_strength']:.10f}\n"
            )
        f.write("\n")

        f.write("Top 10 strongest positive edges\n")
        f.write("-------------------------------\n")
        for _, row in top_positive_edges.head(10).iterrows():
            f.write(f"{row['source']} -> {row['target']}: {row['weight']:.10f}\n")

    print(f"Saved summary: {OUTPUT_SUMMARY}")


def print_top_results(metrics, top_positive_edges):
    print("\nTop 10 by signed degree:")
    print(
        metrics.sort_values("signed_degree", ascending=False)
        [["stock", "signed_degree"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 by absolute out-strength:")
    print(
        metrics.sort_values("abs_out_strength", ascending=False)
        [["stock", "abs_out_strength", "positive_out_strength", "negative_out_strength"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 strongest positive edges:")
    print(
        top_positive_edges[["source", "target", "weight"]]
        .head(10)
        .to_string(index=False)
    )


def main():
    A, stock_names = load_data()

    metrics = compute_node_metrics(A, stock_names)
    _, top_positive_edges, top_abs_edges = compute_edge_metrics(A, stock_names, TOP_EDGES)

    metrics.to_csv(OUTPUT_NODE_METRICS, index=False)
    top_positive_edges.to_csv(OUTPUT_TOP_EDGES, index=False)
    top_abs_edges.to_csv(OUTPUT_TOP_ABS_EDGES, index=False)

    save_graph_summary(A, metrics, top_positive_edges)
    make_plots(A, metrics)
    print_top_results(metrics, top_positive_edges)

    print("\nSaved CSV files:")
    print(f"  {OUTPUT_NODE_METRICS}")
    print(f"  {OUTPUT_TOP_EDGES}")
    print(f"  {OUTPUT_TOP_ABS_EDGES}")


if __name__ == "__main__":
    main()
