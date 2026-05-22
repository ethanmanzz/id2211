"""
Compare CASR Stage 2 on com-Amazon:
  - local acceptance only
  - local + global modularity guard (no decrease)
"""

import time

import pandas as pd

from simple_casr_experiment import (
    load_edges,
    load_amazon_labels_from_communities,
    run_louvain,
    stage1_consensus_partition,
    stage2_refinement,
    evaluate_partition,
)

R = 20
TAU = 0.3
ALPHA = 3.0
MIN_SIZE = 10
STABILITY_THRESHOLD = 0.99
SEED = 0


def main():
    print("Loading com-Amazon...")
    G = load_edges("com-amazon.ungraph.txt")
    labels = load_amazon_labels_from_communities("com-amazon.top5000.cmty.txt")

    rows = []

    print("\nBaseline Louvain...")
    t0 = time.time()
    baseline = run_louvain(G, seed=SEED)
    ev = evaluate_partition(G, baseline, labels)
    rows.append({
        "method": "Baseline Louvain",
        "global_modularity_guard": None,
        "NMI": ev["NMI"],
        "ARI": ev["ARI"],
        "modularity": ev["modularity"],
        "num_communities": ev["num_communities"],
        "refinement_candidates": None,
        "communities_refined": None,
        "runtime_seconds": time.time() - t0,
    })

    print("CASR Stage 1...")
    t0 = time.time()
    coarse, consensus, _ = stage1_consensus_partition(
        G, R=R, tau=TAU, seed=SEED
    )
    stage1_time = time.time() - t0
    ev1 = evaluate_partition(G, coarse, labels)
    rows.append({
        "method": "CASR Stage 1 only",
        "global_modularity_guard": None,
        "NMI": ev1["NMI"],
        "ARI": ev1["ARI"],
        "modularity": ev1["modularity"],
        "num_communities": ev1["num_communities"],
        "refinement_candidates": None,
        "communities_refined": None,
        "runtime_seconds": stage1_time,
    })

    variants = [
        (False, 0.0, "CASR Stage 2 (local only)"),
        (True, 0.0, "CASR Stage 2 (+ global Q guard, strict)"),
        (True, 0.001, "CASR Stage 2 (+ global Q guard, tol=0.001)"),
    ]

    for guard, tol, label in variants:
        print(f"\n{label}...")
        t0 = time.time()
        final, cand, ref = stage2_refinement(
            G,
            coarse,
            consensus,
            alpha=ALPHA,
            min_size=MIN_SIZE,
            stability_threshold=STABILITY_THRESHOLD,
            seed=SEED,
            require_global_modularity=guard,
            global_modularity_tolerance=tol,
        )
        elapsed = time.time() - t0
        ev = evaluate_partition(G, final, labels)
        rows.append({
            "method": label,
            "global_modularity_guard": guard,
            "global_modularity_tolerance": tol if guard else None,
            "NMI": ev["NMI"],
            "ARI": ev["ARI"],
            "modularity": ev["modularity"],
            "num_communities": ev["num_communities"],
            "refinement_candidates": cand,
            "communities_refined": ref,
            "runtime_seconds": stage1_time + elapsed,
        })
        print(
            f"  candidates={cand}, refined={ref}, "
            f"NMI={ev['NMI']:.4f}, ARI={ev['ARI']:.4f}, "
            f"Q={ev['modularity']:.4f}, communities={ev['num_communities']}"
        )

    df = pd.DataFrame(rows)
    out = "compare_amazon_global_guard.csv"
    df.to_csv(out, index=False)

    print(f"\n=== Final comparison (threshold={STABILITY_THRESHOLD}) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
