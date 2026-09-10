"""
Report mean +/- std across random seeds instead of a single run.

WHY THIS EXISTS
    A single training run's metric is one sample from a distribution. Quoting
    it as "the" result invites cherry-picking, and small differences between
    model variants are often smaller than seed noise -- so an ablation that
    compares one run against one run can easily be measuring nothing.

    Reporting mean +/- std over several seeds makes the comparison honest:
    if the gap between two variants is inside the spread, it is not a result.

    Training is pinned to deterministic single-thread mode, so each seed is
    exactly reproducible; the spread here is genuine initialisation and
    shuffling variance, not floating-point drift.

    python scripts/multi_seed.py [n_seeds]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import build_dataset                                  # noqa: E402
from two_tower import (                                          # noqa: E402
    train_two_tower, score_all_items, rank_of_targets,
    ranking_metrics, popularity_baseline,
)
from rerank import train_reranker                                # noqa: E402

METRICS = ["Recall@10", "Recall@50", "NDCG@10", "MRR"]


def summarise(name, runs):
    line = f"{name:<24}"
    for k in METRICS:
        vals = np.array([r[k] for r in runs])
        line += f"  {vals.mean():.4f}+/-{vals.std():.4f}"
    return line


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ds = build_dataset()
    tu, ti = ds.test.user.to_numpy(), ds.test.item.to_numpy()

    nolq, withlq, reranked = [], [], []

    for seed in range(n_seeds):
        print(f"seed {seed}…", flush=True)

        m0 = train_two_tower(ds, epochs=30, seed=seed,
                             logq_correction=False, verbose=False)
        nolq.append(ranking_metrics(rank_of_targets(
            score_all_items(m0, ds, mask_val=True), tu, ti)))

        m1 = train_two_tower(ds, epochs=30, seed=seed,
                             logq_correction=True, verbose=False)
        withlq.append(ranking_metrics(rank_of_targets(
            score_all_items(m1, ds, mask_val=True), tu, ti)))

        reranked.append(train_reranker(ds, m1)["reranked"])

        print(f"  no-logQ NDCG@10 {nolq[-1]['NDCG@10']:.4f} | "
              f"+logQ {withlq[-1]['NDCG@10']:.4f} | "
              f"+rerank {reranked[-1]['NDCG@10']:.4f}", flush=True)

    pop = popularity_baseline(ds, tu, ti)

    print("\n" + "=" * 86)
    print(f"MEAN +/- STD over {n_seeds} seeds")
    print("=" * 86)
    print(f"{'Model':<24}" + "".join(f"  {k:^15}" for k in METRICS))
    print(summarise("1. Popularity baseline", [pop]))       # deterministic
    print(summarise("2. Two-tower (no logQ)", nolq))
    print(summarise("3. Two-tower + logQ", withlq))
    print(summarise("4. + LightGBM re-rank", reranked))

    a = np.array([r["NDCG@10"] for r in nolq])
    b = np.array([r["NDCG@10"] for r in withlq])
    c = np.array([r["NDCG@10"] for r in reranked])
    print(f"\nlogQ correction : {(b.mean()/a.mean()-1)*100:+.1f}% NDCG@10 "
          f"(spread {a.std():.4f} / {b.std():.4f})")
    print(f"re-ranking      : {(c.mean()/b.mean()-1)*100:+.1f}% NDCG@10")
    print(f"vs baseline     : {(c.mean()/pop['NDCG@10']-1)*100:+.1f}% NDCG@10")


if __name__ == "__main__":
    main()
