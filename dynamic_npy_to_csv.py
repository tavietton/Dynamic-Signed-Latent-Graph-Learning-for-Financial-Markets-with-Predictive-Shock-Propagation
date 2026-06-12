import os
import numpy as np
import pandas as pd


ADJ_FILE = "dynamic_adjacency_mean.npy"
META_FILE = "meta.npz"

OUTPUT_ADJ_CSV = "dynamic_adjacency_mean.csv"
OUTPUT_TOP_INFLUENTIAL = "dynamic_top_influential_stocks_full.csv"
OUTPUT_TOP_EDGES = "dynamic_top_edges_500.csv"
OUTPUT_GRAPH_SUMMARY = "dynamic_graph_summary.txt"

TOP_EDGES = 500


def load_files():
    if not os.path.exists(ADJ_FILE):
        raise FileNotFoundError(f"Cannot find {ADJ_FILE}")

    A = np.load(ADJ_FILE)

    if os.path.exists(META_FILE):
        meta = np.load(META_FILE, allow_pickle=True)

        if "stock_names" in meta.files:
            stock_names = meta["stock_names"]
        else:
            stock_names = np.array([f"Stock_{i}" for i in range(A.shape[0])])
    else:
        stock_names = np.array([f"Stock_{i}" for i in range(A.shape[0])])

    stock_names = np.array(stock_names).astype(str)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    if len(stock_names) != A.shape[0]:
        raise ValueError("Number of stock names does not match adjacency size.")

    print("Loaded adjacency:", A.shape)
    print("Loaded stock names:", len(stock_names))

    return A, stock_names


def save_adjacency_csv(A, stock_names):
    A_df = pd.DataFrame(A, index=stock_names, columns=stock_names)
    A_df.to_csv(OUTPUT_ADJ_CSV)

    print(f"Saved: {OUTPUT_ADJ_CSV}")


def save_top_influential(A, stock_names):
    signed_degree = A.sum(axis=1)
    abs_out_strength = np.abs(A).sum(axis=1)
    positive_out_strength = np.maximum(A, 0).sum(axis=1)
    negative_out_strength = np.abs(np.minimum(A, 0)).sum(axis=1)

    df = pd.DataFrame({
        "stock": stock_names,
        "signed_degree": signed_degree,
        "abs_out_strength": abs_out_strength,
        "positive_out_strength": positive_out_strength,
        "negative_out_strength": negative_out_strength,
    })

    df = df.sort_values("signed_degree", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_TOP_INFLUENTIAL, index=False)

    print(f"Saved: {OUTPUT_TOP_INFLUENTIAL}")

    return df


def save_top_edges(A, stock_names):
    edges = []
    N = A.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            edges.append({
                "source": stock_names[i],
                "target": stock_names[j],
                "weight": A[i, j],
                "abs_weight": abs(A[i, j]),
            })

    edges_df = pd.DataFrame(edges)

    edges_df = edges_df.sort_values("weight", ascending=False).reset_index(drop=True)
    edges_df.head(TOP_EDGES).to_csv(OUTPUT_TOP_EDGES, index=False)

    print(f"Saved: {OUTPUT_TOP_EDGES}")

    return edges_df


def save_summary(A, node_df):
    signed_degree = A.sum(axis=1)
    abs_out_strength = np.abs(A).sum(axis=1)
    positive_out_strength = np.maximum(A, 0).sum(axis=1)
    negative_out_strength = np.abs(np.minimum(A, 0)).sum(axis=1)

    with open(OUTPUT_GRAPH_SUMMARY, "w") as f:
        f.write("Dynamic Graph Summary\n")
        f.write("=====================\n")
        f.write(f"Adjacency file: {ADJ_FILE}\n")
        f.write(f"Adjacency shape: {A.shape}\n")
        f.write(f"Number of stocks: {A.shape[0]}\n")
        f.write("\n")

        f.write("Edge weights\n")
        f.write("------------\n")
        f.write(f"Min weight: {A.min()}\n")
        f.write(f"Max weight: {A.max()}\n")
        f.write(f"Mean weight: {A.mean()}\n")
        f.write(f"Std weight: {A.std()}\n")
        f.write("\n")

        f.write("Signed degree\n")
        f.write("-------------\n")
        f.write(f"Min signed degree: {signed_degree.min()}\n")
        f.write(f"Max signed degree: {signed_degree.max()}\n")
        f.write(f"Mean signed degree: {signed_degree.mean()}\n")
        f.write(f"Std signed degree: {signed_degree.std()}\n")
        f.write("\n")

        f.write("Absolute out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min abs out-strength: {abs_out_strength.min()}\n")
        f.write(f"Max abs out-strength: {abs_out_strength.max()}\n")
        f.write(f"Mean abs out-strength: {abs_out_strength.mean()}\n")
        f.write(f"Std abs out-strength: {abs_out_strength.std()}\n")
        f.write("\n")

        f.write("Positive out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min positive out-strength: {positive_out_strength.min()}\n")
        f.write(f"Max positive out-strength: {positive_out_strength.max()}\n")
        f.write(f"Mean positive out-strength: {positive_out_strength.mean()}\n")
        f.write(f"Std positive out-strength: {positive_out_strength.std()}\n")
        f.write("\n")

        f.write("Negative out-strength\n")
        f.write("---------------------\n")
        f.write(f"Min negative out-strength: {negative_out_strength.min()}\n")
        f.write(f"Max negative out-strength: {negative_out_strength.max()}\n")
        f.write(f"Mean negative out-strength: {negative_out_strength.mean()}\n")
        f.write(f"Std negative out-strength: {negative_out_strength.std()}\n")
        f.write("\n")

        f.write("Top 10 by signed degree\n")
        f.write("-----------------------\n")
        top_signed = node_df.sort_values("signed_degree", ascending=False).head(10)
        for _, row in top_signed.iterrows():
            f.write(f"{row['stock']}: {row['signed_degree']}\n")

        f.write("\nTop 10 by absolute out-strength\n")
        f.write("--------------------------------\n")
        top_abs = node_df.sort_values("abs_out_strength", ascending=False).head(10)
        for _, row in top_abs.iterrows():
            f.write(
                f"{row['stock']}: "
                f"abs={row['abs_out_strength']}, "
                f"pos={row['positive_out_strength']}, "
                f"neg={row['negative_out_strength']}\n"
            )

    print(f"Saved: {OUTPUT_GRAPH_SUMMARY}")


def main():
    A, stock_names = load_files()

    save_adjacency_csv(A, stock_names)
    node_df = save_top_influential(A, stock_names)
    save_top_edges(A, stock_names)
    save_summary(A, node_df)

    print("\nDone.")
    print("Saved files:")
    print(f"  {OUTPUT_ADJ_CSV}")
    print(f"  {OUTPUT_TOP_INFLUENTIAL}")
    print(f"  {OUTPUT_TOP_EDGES}")
    print(f"  {OUTPUT_GRAPH_SUMMARY}")


if __name__ == "__main__":
    main()
