
"""
Goal:
    Compare baseline Louvain vs. your proposed CASR algorithm.

Outputs:
    - NMI and ARI, if ground truth is provided
    - Modularity
    - Runtime
    - Stability of baseline Louvain across repeated runs
    - A CSV file with final comparison results

Install:
    pip install networkx numpy pandas scikit-learn

To run on either dataset, use the following in the terminal:

1. email-Eu-core:
    python simple_casr_experiment.py \
  --dataset email \
  --edges email-Eu-core.txt \
  --labels email-Eu-core-department-labels.txt \
  --R 20 \
  --tau 0.3 \
  --alpha 3.0 \
  --min-size 10 \
  --stability-threshold 0.95

2. com-Amazon:
    python simple_casr_experiment.py \
  --dataset amazon \
  --edges com-amazon.ungraph.txt \
  --communities com-amazon.top5000.cmty.txt \
  --R 10 \
  --tau 0.3 \
  --alpha 3.0 \
  --min-size 50 \
  --stability-threshold 0.95
"""

import argparse
import time
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ============================================================
# 1. Load data
# ============================================================

def load_edges(path):
    """Load an undirected edge list."""
    G = nx.Graph()

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            u, v = map(int, line.split()[:2])

            if u != v:
                G.add_edge(u, v)

    return G


def load_email_labels(path):
    """
    Load email-Eu-core department labels.

    Format:
        node_id department_id
    """
    labels = {}

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            node, label = map(int, line.split()[:2])
            labels[node] = label

    return labels


def load_amazon_labels_from_communities(path):
    """
    Load com-Amazon ground-truth communities.

    Important:
        Amazon communities are overlapping.
        NMI/ARI require one label per node.
        To keep this script simple, each node is assigned to the first
        community in which it appears.
    """
    labels = {}

    with open(path, "r") as f:
        for cid, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue

            nodes = list(map(int, line.split()))

            for node in nodes:
                if node not in labels:
                    labels[node] = cid

    return labels


# ============================================================
# 2. Basic partition helpers
# ============================================================

def communities_to_partition(communities):
    """Convert list of node sets into node -> community_id."""
    partition = {}

    for cid, community in enumerate(communities):
        for node in community:
            partition[node] = cid

    return partition


def partition_to_communities(partition):
    """Convert node -> community_id into list of node sets."""
    groups = defaultdict(set)

    for node, cid in partition.items():
        groups[cid].add(node)

    return list(groups.values())


def relabel_partition(partition):
    """Relabel community IDs to 0, 1, 2, ..."""
    mapping = {}
    new_partition = {}
    next_id = 0

    for node in sorted(partition):
        old = partition[node]

        if old not in mapping:
            mapping[old] = next_id
            next_id += 1

        new_partition[node] = mapping[old]

    return new_partition


# ============================================================
# 3. Louvain baseline
# ============================================================

def run_louvain(G, seed=0, resolution=1.0, weight="weight"):
    """Run Louvain and return node -> community_id."""
    communities = nx.community.louvain_communities(
        G,
        seed=seed,
        resolution=resolution,
        weight=weight
    )

    return communities_to_partition(communities)


def run_louvain_many_times(G, R=10, seed=0):
    """Run Louvain multiple times using different seeds."""
    rng = random.Random(seed)
    partitions = []

    for _ in range(R):
        s = rng.randint(0, 10**9)
        partitions.append(run_louvain(G, seed=s))

    return partitions


# ============================================================
# 4. Stage 1: Consensus-Based Stable Partition
# ============================================================

def build_edge_consensus(G, partitions):
    """
    Build consensus only on original graph edges.

    For each edge (u, v), M_uv is the fraction of Louvain runs
    where u and v are assigned to the same community.

    This is much simpler and more scalable than building a full n x n matrix.
    """
    R = len(partitions)
    consensus = {}

    for u, v in G.edges():
        same_count = 0

        for p in partitions:
            if p[u] == p[v]:
                same_count += 1

        consensus[(u, v)] = same_count / R

    return consensus


def build_consensus_graph(G, consensus, tau=0.5):
    """
    Build the uncertainty-filtered consensus graph.

    Keep edge (u, v) only if M_uv >= tau.
    Use M_uv as the edge weight.
    """
    Gc = nx.Graph()
    Gc.add_nodes_from(G.nodes())

    for (u, v), mij in consensus.items():
        if mij >= tau:
            Gc.add_edge(u, v, weight=mij)

    return Gc


def stage1_consensus_partition(G, R=10, tau=0.5, seed=0):
    """
    Stage 1:
        1. Run Louvain R times.
        2. Build edge-based consensus values.
        3. Build consensus graph.
        4. Run weighted Louvain on the consensus graph.
    """
    partitions = run_louvain_many_times(G, R=R, seed=seed)
    consensus = build_edge_consensus(G, partitions)
    Gc = build_consensus_graph(G, consensus, tau=tau)
    coarse_partition = run_louvain(Gc, seed=seed + 1, weight="weight")

    return coarse_partition, consensus, partitions


# ============================================================
# 5. Stage 2: Instability-Aware Local Refinement
# ============================================================

def community_stability(G, nodes, consensus):
    """
    Compute s(C): average consensus value over internal edges.
    """
    sub_edges = list(G.subgraph(nodes).edges())

    if len(sub_edges) == 0:
        return 1.0

    values = []

    for u, v in sub_edges:
        if (u, v) in consensus:
            values.append(consensus[(u, v)])
        else:
            values.append(consensus.get((v, u), 0.0))

    return float(np.mean(values))


def stage2_refinement(
    G,
    coarse_partition,
    consensus,
    alpha=2.0,
    min_size=10,
    stability_threshold=0.50,
    seed=0
):
    """
    Stage 2:
        Refine only large communities with low stability.

    A community is refined if:
        size >= min_size
        s(C) < stability_threshold

    Local Louvain uses:
        gamma_C = 1 + alpha * (1 - s(C))
    """
    final_partition = dict(coarse_partition)
    communities = partition_to_communities(coarse_partition)

    next_cid = max(final_partition.values()) + 1
    num_candidates = 0
    num_refined = 0

    for idx, C in enumerate(communities):
        C = set(C)

        if len(C) < min_size:
            continue

        s_C = community_stability(G, C, consensus)

        if s_C >= stability_threshold:
            continue

        num_candidates += 1

        G_sub = G.subgraph(C).copy()

        if G_sub.number_of_edges() == 0:
            continue

        gamma_C = 1.0 + alpha * (1.0 - s_C)

        refined = run_louvain(
            G_sub,
            seed=seed + idx + 100,
            resolution=gamma_C,
            weight="weight"
        )

        refined_communities = partition_to_communities(refined)

        # Only accept if it actually splits the community.
        if len(refined_communities) <= 1:
            continue

        num_refined += 1

        for sub_C in refined_communities:
            for node in sub_C:
                final_partition[node] = next_cid
            next_cid += 1

    final_partition = relabel_partition(final_partition)

    return final_partition, num_candidates, num_refined

def run_casr_once(G, R=10, tau=0.5, alpha=1.0, min_size=30,
                  stability_threshold=0.95, seed=0):
    coarse_partition, consensus, _ = stage1_consensus_partition(
        G,
        R=R,
        tau=tau,
        seed=seed
    )

    final_partition, _, _ = stage2_refinement(
        G,
        coarse_partition,
        consensus,
        alpha=alpha,
        min_size=min_size,
        stability_threshold=stability_threshold,
        seed=seed
    )

    return final_partition

def casr_stability_score(G, num_runs=5, R=10, tau=0.5, alpha=1.0,
                         min_size=30, stability_threshold=0.95, seed=0):
    partitions = []

    for i in range(num_runs):
        p = run_casr_once(
            G,
            R=R,
            tau=tau,
            alpha=alpha,
            min_size=min_size,
            stability_threshold=stability_threshold,
            seed=seed + i * 100
        )
        partitions.append(p)

    return stability_score(partitions, sorted(G.nodes()))

# ============================================================
# 6. Evaluation
# ============================================================

def evaluate_partition(G, partition, labels=None):
    """
    Evaluate one partition.

    NMI and ARI are computed only if labels are available.
    """
    communities = partition_to_communities(partition)

    result = {
        "num_communities": len(communities),
        "modularity": nx.community.modularity(G, communities),
    }

    if labels is not None:
        common_nodes = sorted(set(partition) & set(labels))

        y_true = [labels[n] for n in common_nodes]
        y_pred = [partition[n] for n in common_nodes]

        result["NMI"] = normalized_mutual_info_score(y_true, y_pred)
        result["ARI"] = adjusted_rand_score(y_true, y_pred)
        result["labeled_nodes"] = len(common_nodes)
    else:
        result["NMI"] = np.nan
        result["ARI"] = np.nan
        result["labeled_nodes"] = 0

    return result


def stability_score(partitions, nodes):
    """
    Measure algorithm stability.

    We compute average pairwise NMI between different Louvain runs.
    Higher value means more stable.
    """
    scores = []

    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a = [partitions[i][n] for n in nodes]
            b = [partitions[j][n] for n in nodes]
            scores.append(normalized_mutual_info_score(a, b))

    return float(np.mean(scores))


# ============================================================
# 7. Main experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["email", "amazon"], required=True)
    parser.add_argument("--edges", required=True)

    parser.add_argument("--labels", default=None, help="Department labels for email-Eu-core")
    parser.add_argument("--communities", default=None, help="Ground-truth communities for com-Amazon")

    parser.add_argument("--R", type=int, default=10, help="Number of Louvain runs for consensus")
    parser.add_argument("--tau", type=float, default=0.5, help="Consensus threshold")
    parser.add_argument("--alpha", type=float, default=1.0, help="Refinement strength")
    parser.add_argument("--min-size", type=int, default=30, help="Minimum community size to refine")
    parser.add_argument("--stability-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # -----------------------------
    # Load graph and labels
    # -----------------------------
    print("Loading graph...")
    G = load_edges(args.edges)

    labels = None

    if args.dataset == "email" and args.labels:
        labels = load_email_labels(args.labels)

    if args.dataset == "amazon" and args.communities:
        labels = load_amazon_labels_from_communities(args.communities)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # -----------------------------
    # Baseline Louvain
    # -----------------------------
    print("\nRunning baseline Louvain...")
    start = time.time()
    baseline_partition = run_louvain(G, seed=args.seed)
    baseline_time = time.time() - start

    baseline_result = evaluate_partition(G, baseline_partition, labels)
    baseline_result["method"] = "Baseline Louvain"
    baseline_result["runtime_seconds"] = baseline_time

    # Stability of baseline Louvain
    print("Measuring Louvain stability...")
    baseline_runs = run_louvain_many_times(G, R=args.R, seed=args.seed)
    baseline_result["stability_pairwise_NMI"] = stability_score(baseline_runs, sorted(G.nodes()))

    # -----------------------------
    # CASR Stage 1 + Stage 2
    # -----------------------------
    print("\nRunning CASR Stage 1 and Stage 2...")
    start = time.time()

    coarse_partition, consensus, _ = stage1_consensus_partition(
        G,
        R=args.R,
        tau=args.tau,
        seed=args.seed
    )

    final_partition, num_candidates, num_refined = stage2_refinement(
        G,
        coarse_partition,
        consensus,
        alpha=args.alpha,
        min_size=args.min_size,
        stability_threshold=args.stability_threshold,
        seed=args.seed
    )

    casr_time = time.time() - start

    casr_result = evaluate_partition(G, final_partition, labels)
    casr_result["method"] = "CASR"
    casr_result["runtime_seconds"] = casr_time
    #casr_result["stability_pairwise_NMI"] = np.nan
    print("Measuring CASR stability...")
    casr_result["stability_pairwise_NMI"] = casr_stability_score(
        G,
        num_runs=5,
        R=args.R,
        tau=args.tau,
        alpha=args.alpha,
        min_size=args.min_size,
        stability_threshold=args.stability_threshold,
        seed=args.seed
        )
    
    casr_result["refinement_candidates"] = num_candidates
    casr_result["communities_refined"] = num_refined

    baseline_result["refinement_candidates"] = np.nan
    baseline_result["communities_refined"] = np.nan

    # -----------------------------
    # Save and print results
    # -----------------------------
    results = pd.DataFrame([baseline_result, casr_result])

    columns = [
        "method",
        "NMI",
        "ARI",
        "modularity",
        "num_communities",
        "runtime_seconds",
        "stability_pairwise_NMI",
        "refinement_candidates",
        "communities_refined",
        "labeled_nodes",
    ]

    results = results[columns]

    output_file = f"results_{args.dataset}.csv"
    results.to_csv(output_file, index=False)

    print("\nFinal comparison:")
    print(results)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
