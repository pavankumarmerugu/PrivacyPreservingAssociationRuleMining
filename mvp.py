#!/usr/bin/env python3
"""
Privacy-Preserving Association Rule Mining — MVP
=================================================
Pipeline:
  1. Load dataset (Deezer or Last.fm)
  2. Encode as a boolean user-item binary matrix
  3. Build item-item cosine similarity matrix
  4. Form interest groups via centroid-based similarity
  5. Mine frequent itemsets with the Apriori algorithm
  6. Generate recommendations for sampled users
  7. Evaluate with Precision & Recall

Usage:
  python mvp.py                          # Deezer dataset, default settings
  python mvp.py --dataset lastfm         # Last.fm dataset
  python mvp.py --min-support 0.15 --n-users 20
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def load_deezer(path: str) -> pd.DataFrame:
    """Load RO_genres.json and return a boolean user-genre binary matrix."""
    with open(path) as f:
        raw = json.load(f)
    items = list(raw.values())
    te = TransactionEncoder()
    matrix = te.fit(items).transform(items)
    return pd.DataFrame(matrix, columns=te.columns_)


def load_lastfm(path: str) -> pd.DataFrame:
    """Load lastfm.csv and return a boolean user-item binary matrix."""
    data = pd.read_csv(path)
    return data.drop("user", axis=1).astype(bool)


# ── 2. Interest Group Formation ───────────────────────────────────────────────

def build_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the item-item cosine similarity matrix."""
    float_df = df.astype(float)
    magnitude = np.sqrt(np.square(float_df).sum(axis=1)).replace(0, 1)
    normalized = float_df.divide(magnitude, axis="index")
    sim = cosine_similarity(sparse.csr_matrix(normalized).T)
    return pd.DataFrame(sim, index=df.columns, columns=df.columns)


def form_interest_groups(
    df: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    n_groups: int = 5,
) -> tuple:
    """
    Select n_groups evenly-spaced centroid items and return each group's
    most-similar neighbours.

    Returns:
        centroids  – list of centroid item names
        groups     – list of lists of neighbouring item names
    """
    items = df.columns.tolist()
    n_items = len(items)
    step = max(1, n_items // (n_groups + 1))
    group_size = step

    centroids = [items[step * (i + 1)] for i in range(min(n_groups, n_items - 1))]
    groups = [
        sim_matrix.loc[c].nlargest(group_size).index.tolist()
        for c in centroids
    ]
    return centroids, groups


# ── 3. Frequent Itemset Mining ────────────────────────────────────────────────

def mine_frequent_itemsets(
    df: pd.DataFrame,
    min_support: float,
    max_len: int,
) -> pd.DataFrame:
    """Run Apriori and return a DataFrame of frequent itemsets."""
    return apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)


# ── 4. Recommendation & Evaluation ───────────────────────────────────────────

def recommend_for_user(user_row: pd.Series, itemsets_df: pd.DataFrame) -> list:
    """
    Association-rule recommendation: for each 2-itemset {A, B}, if A is in the
    user's likes, add {A, B} to the recommendation list (and vice versa).

    Returns a list of (itemset, support) tuples for itemsets triggered by the
    user's profile.
    """
    user_likes = set(user_row[user_row].index.tolist())
    triggered = []
    for _, row in itemsets_df.iterrows():
        itemset = row["itemsets"]
        if len(itemset) == 2:
            a, b = list(itemset)
            if a in user_likes or b in user_likes:
                triggered.append((itemset, row["support"]))
    return triggered


def evaluate(user_likes: set, triggered_rules: list) -> tuple:
    """
    Evaluate association-rule coverage against the user's full profile.

    For each triggered 2-itemset, precision = fraction of that itemset covered
    by the user's likes. Macro-averages are returned.

    Returns:
        precision – average per-rule precision
        recall    – fraction of the user's liked items covered by any triggered rule
    """
    if not triggered_rules or not user_likes:
        return 0.0, 0.0

    rule_precisions = []
    covered_likes = set()
    for itemset, _ in triggered_rules:
        itemset_set = set(itemset)
        rule_precisions.append(len(user_likes & itemset_set) / len(itemset_set))
        covered_likes |= user_likes & itemset_set

    precision = float(np.mean(rule_precisions))
    recall = len(covered_likes) / len(user_likes)
    return precision, recall


# ── 5. Pipeline Orchestration ─────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    min_support: float,
    max_len: int,
    n_groups: int,
    n_users: int,
    seed: int,
) -> dict:
    """
    Execute the full Privacy-Preserving Association Rule Mining pipeline.

    Returns a dict with average precision and recall.
    """
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Dataset : {df.shape[0]:,} users  ×  {df.shape[1]:,} items")
    print(f"  Settings: min_support={min_support}, max_len={max_len}, "
          f"n_groups={n_groups}, n_users={n_users}")
    print(sep)

    # ── Step 1: Similarity matrix ─────────────────────────────────────────────
    print("\n[Step 1/4]  Building item-item cosine similarity matrix …")
    t0 = time.time()
    sim_matrix = build_similarity_matrix(df)
    print(f"            Done in {time.time() - t0:.2f}s  "
          f"({sim_matrix.shape[0]}×{sim_matrix.shape[1]} matrix)")

    # ── Step 2: Interest groups ───────────────────────────────────────────────
    print(f"\n[Step 2/4]  Forming {n_groups} interest groups …")
    centroids, groups = form_interest_groups(df, sim_matrix, n_groups)
    for i, (centroid, group) in enumerate(zip(centroids, groups)):
        preview = ", ".join(group[:4])
        suffix = " …" if len(group) > 4 else ""
        print(f"  Group {i + 1:2d}  [{centroid}]  →  {preview}{suffix}")

    # Jaccard similarity between adjacent groups
    if len(groups) > 1:
        print("\n  Jaccard similarities between adjacent groups:")
        for i in range(len(groups) - 1):
            a, b = set(groups[i]), set(groups[i + 1])
            jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
            print(f"    Group {i + 1} ∩ Group {i + 2} : {jaccard:.4f}")

    # ── Step 3: Frequent itemsets ─────────────────────────────────────────────
    print(f"\n[Step 3/4]  Mining frequent itemsets "
          f"(min_support={min_support}, max_len={max_len}) …")
    t0 = time.time()
    itemsets_df = mine_frequent_itemsets(df, min_support, max_len)
    elapsed = time.time() - t0
    print(f"            Found {len(itemsets_df):,} frequent itemsets in {elapsed:.2f}s")

    if itemsets_df.empty:
        print("\n  ⚠  No frequent itemsets found — try lowering --min-support.")
        return {"avg_precision": 0.0, "avg_recall": 0.0}

    # Show top 5 by support
    print("\n  Top frequent itemsets by support:")
    top = itemsets_df.nlargest(5, "support")
    for _, row in top.iterrows():
        print(f"    support={row['support']:.4f}  {set(row['itemsets'])}")

    # ── Step 4: Recommendations & evaluation ─────────────────────────────────
    print(f"\n[Step 4/4]  Generating recommendations for {n_users} sampled users …")
    random.seed(seed)
    sample_indices = random.sample(range(len(df)), min(n_users, len(df)))

    precisions, recalls = [], []
    print(f"\n  {'User':>7}  {'Likes':>5}  {'Rules':>5}  {'Precision':>10}  {'Recall':>8}")
    print("  " + "-" * 44)

    for idx in sample_indices:
        user_row = df.iloc[idx]
        user_likes = set(user_row[user_row].index.tolist())
        triggered = recommend_for_user(user_row, itemsets_df)
        p, r = evaluate(user_likes, triggered)
        precisions.append(p)
        recalls.append(r)
        print(f"  {idx:>7,}  {len(user_likes):>5}  {len(triggered):>5}  "
              f"{p:>10.4f}  {r:>8.4f}")

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))

    print("  " + "-" * 44)
    print(f"  {'Average':>7}               "
          f"  {avg_p:>10.4f}  {avg_r:>8.4f}")
    print(f"\n{sep}\n")

    return {"avg_precision": avg_p, "avg_recall": avg_r}


# ── CLI Entry-Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Association Rule Mining MVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["deezer", "lastfm"],
        default="deezer",
        help="Dataset to use",
    )
    parser.add_argument(
        "--deezer-path",
        default="RO_genres.json",
        help="Path to the Deezer RO_genres.json file",
    )
    parser.add_argument(
        "--lastfm-path",
        default="lastfm.csv",
        help="Path to the Last.fm lastfm.csv file",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="Minimum support for Apriori (default: 0.2 for deezer, 0.1 for lastfm)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=5,
        help="Maximum itemset length for Apriori",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="Number of interest groups to form",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=10,
        help="Number of users to sample for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.dataset == "deezer":
        min_support = args.min_support if args.min_support is not None else 0.2
        print(f"Loading Deezer dataset from '{args.deezer_path}' …")
        df = load_deezer(args.deezer_path)
    else:
        min_support = args.min_support if args.min_support is not None else 0.02
        print(f"Loading Last.fm dataset from '{args.lastfm_path}' …")
        df = load_lastfm(args.lastfm_path)

    run_pipeline(
        df,
        min_support=min_support,
        max_len=args.max_len,
        n_groups=args.n_groups,
        n_users=args.n_users,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
