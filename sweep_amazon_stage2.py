"""
Sweep stability_threshold and min_size on com-Amazon (Stage 2 activation).
Reuses a single Stage-1 consensus partition to save runtime.
"""

import time
from itertools import product

import pandas as pd

from simple_casr_experiment import (
    load_edges,
    load_amazon_labels_from_communities,
    run_louvain,
    stage1_consensus_partition,
    stage2_refinement,
    evaluate_partition,
)

# Fixed CASR Stage-1 settings (match notebook / proposal defaults)
EDGE_FILE = "com-amazon.ungraph.txt"
COMMUNITY_FILE = "com-amazon.top5000.cmty.txt"
R = 20
TAU = 0.3
ALPHA = 3.0
SEED = 0

STABILITY_THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75]
MIN_SIZES = [10, 20, 30, 50, 100]


def main():
    print("Loading com-Amazon...")
    G = load_edges(EDGE_FILE)
    labels = load_amazon_labels_from_communities(COMMUNITY_FILE)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("\nBaseline Louvain (single run)...")
    t0 = time.time()
    baseline = run_louvain(G, seed=SEED)
    baseline_eval = evaluate_partition(G, baseline, labels)
    baseline_eval["method"] = "baseline_louvain"
    baseline_eval["stability_threshold"] = None
    baseline_eval["min_size"] = None
    baseline_eval["refinement_candidates"] = None
    baseline_eval["communities_refined"] = None
    baseline_eval["stage1_seconds"] = None
    baseline_eval["stage2_seconds"] = time.time() - t0
    print(baseline_eval)

    print("\nCASR Stage 1 (shared across sweep)...")
    t0 = time.time()
    coarse_partition, consensus, _ = stage1_consensus_partition(
        G, R=R, tau=TAU, seed=SEED
    )
    stage1_time = time.time() - t0
    print(f"Stage 1 done in {stage1_time:.1f}s")

    stage1_only = evaluate_partition(G, coarse_partition, labels)
    print(
        f"Stage-1-only: NMI={stage1_only['NMI']:.4f}, ARI={stage1_only['ARI']:.4f}, "
        f"modularity={stage1_only['modularity']:.4f}, communities={stage1_only['num_communities']}"
    )

    rows = []
    total = len(STABILITY_THRESHOLDS) * len(MIN_SIZES)

    for i, (thresh, min_size) in enumerate(
        product(STABILITY_THRESHOLDS, MIN_SIZES), start=1
    ):
        print(f"\n[{i}/{total}] stability_threshold={thresh}, min_size={min_size}")
        t0 = time.time()
        final_partition, num_candidates, num_refined = stage2_refinement(
            G,
            coarse_partition,
            consensus,
            alpha=ALPHA,
            min_size=min_size,
            stability_threshold=thresh,
            seed=SEED,
        )
        stage2_time = time.time() - t0

        ev = evaluate_partition(G, final_partition, labels)
        row = {
            "stability_threshold": thresh,
            "min_size": min_size,
            "NMI": ev["NMI"],
            "ARI": ev["ARI"],
            "modularity": ev["modularity"],
            "num_communities": ev["num_communities"],
            "refinement_candidates": num_candidates,
            "communities_refined": num_refined,
            "stage2_seconds": stage2_time,
            "stage1_seconds": stage1_time,
            "delta_NMI_vs_stage1": ev["NMI"] - stage1_only["NMI"],
            "delta_ARI_vs_stage1": ev["ARI"] - stage1_only["ARI"],
            "delta_NMI_vs_baseline": ev["NMI"] - baseline_eval["NMI"],
            "delta_ARI_vs_baseline": ev["ARI"] - baseline_eval["ARI"],
        }
        rows.append(row)
        print(
            f"  candidates={num_candidates}, refined={num_refined}, "
            f"NMI={ev['NMI']:.4f}, ARI={ev['ARI']:.4f}, "
            f"communities={ev['num_communities']}, stage2={stage2_time:.1f}s"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["refinement_candidates", "NMI"],
        ascending=[False, False],
    )

    out = "sweep_amazon_stage2.csv"
    df.to_csv(out, index=False)

    baseline_row = pd.DataFrame([baseline_eval])
    summary_path = "sweep_amazon_stage2_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Baseline Louvain ===\n")
        f.write(baseline_row.to_string(index=False))
        f.write("\n\n=== Stage 1 only (coarse) ===\n")
        f.write(
            f"NMI={stage1_only['NMI']:.6f} ARI={stage1_only['ARI']:.6f} "
            f"modularity={stage1_only['modularity']:.6f} "
            f"communities={stage1_only['num_communities']}\n"
        )
        f.write("\n=== Top configs by NMI (Stage 2 activated) ===\n")
        activated = df[df["communities_refined"] > 0]
        if len(activated) == 0:
            f.write("No configuration activated Stage 2.\n")
        else:
            f.write(
                activated.sort_values("NMI", ascending=False)
                .head(10)
                .to_string(index=False)
            )
        f.write("\n\n=== Best overall NMI ===\n")
        f.write(
            df.sort_values("NMI", ascending=False).head(5).to_string(index=False)
        )

    print(f"\nSaved: {out}")
    print(f"Summary: {summary_path}")
    print("\n--- Top 5 by NMI (any activation) ---")
    print(df.sort_values("NMI", ascending=False).head(5).to_string(index=False))
    print("\n--- Configs with Stage 2 refined > 0 ---")
    act = df[df["communities_refined"] > 0]
    if len(act) == 0:
        print("(none)")
    else:
        print(act.sort_values("NMI", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
