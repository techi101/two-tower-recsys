"""
End-to-end run: data -> two-tower retrieval -> LightGBM re-ranking -> metrics.

    py -3.12 run.py

Everything is evaluated against a popularity baseline, because a recommender
that cannot beat "show everyone the most popular thing" has learned nothing
about individual users.
"""

import sys, time
import numpy as np

sys.path.insert(0, "src")

from data import build_dataset
from two_tower import (
    train_two_tower, score_all_items, rank_of_targets,
    ranking_metrics, popularity_baseline,
)
from rerank import train_reranker


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    t0 = time.time()

    print("\n[1/4] Loading MovieLens-100K")
    ds = build_dataset()
    print(f"      {ds.n_users} users | {ds.n_items} movies | "
          f"{len(ds.train)} train interactions")
    print( "      chronological leave-one-out: last item per user = test, "
           "second-to-last = val")

    test_users = ds.test.user.to_numpy()
    test_items = ds.test.item.to_numpy()

    print("\n[2/4] Training two-tower retrieval model (PyTorch, in-batch negatives)")
    print("      ablation A: WITHOUT logQ sampling-bias correction")
    model_nolq = train_two_tower(ds, epochs=30, logq_correction=False, verbose=False)
    nolq_metrics = ranking_metrics(rank_of_targets(
        score_all_items(model_nolq, ds, mask_val=True), test_users, test_items))
    print(f"      -> NDCG@10 {nolq_metrics['NDCG@10']:.4f}")

    print("      ablation B: WITH logQ sampling-bias correction")
    model = train_two_tower(ds, epochs=30, logq_correction=True)

    print("\n[3/4] Evaluating STAGE 1 (retrieval) on the test item")
    scores = score_all_items(model, ds, mask_val=True)
    tt_metrics = ranking_metrics(rank_of_targets(scores, test_users, test_items))
    pop_metrics = popularity_baseline(ds, test_users, test_items, mask_val=True)

    print("\n[4/4] Training LightGBM LambdaMART re-ranker over top-100 candidates")
    rr = train_reranker(ds, model)

    # ---------------- results ----------------------------------------------
    METRICS = ["Recall@10", "Recall@50", "NDCG@10", "MRR"]

    def fmt(m):
        return [f"{m.get(k, float('nan')):.4f}" for k in METRICS]

    print("\n" + "=" * 68)
    print("RESULTS  (full catalogue, test item, seen items masked)")
    print("=" * 68)

    rows = [
        ["1. Popularity baseline", *fmt(pop_metrics)],
        ["2. Two-tower (no logQ)", *fmt(nolq_metrics)],
        ["3. Two-tower + logQ", *fmt(tt_metrics)],
        ["4. + LightGBM re-rank", *fmt(rr["reranked"])],
    ]
    print_table(rows, ["Model"] + METRICS)

    def pct(a, b):
        return (a - b) / max(b, 1e-9) * 100

    n = lambda m: m["NDCG@10"]
    print(f"\n  logQ correction  (2 -> 3) : NDCG@10 {pct(n(tt_metrics), n(nolq_metrics)):+.1f}%")
    print(f"  re-ranking       (3 -> 4) : NDCG@10 {pct(n(rr['reranked']), n(tt_metrics)):+.1f}%")
    print(f"  full pipeline vs baseline : NDCG@10 {pct(n(rr['reranked']), n(pop_metrics)):+.1f}%"
          f"  |  Recall@50 {pct(rr['reranked']['Recall@50'], pop_metrics['Recall@50']):+.1f}%")
    print(f"\n  candidate recall@{rr['n_candidates']} = {rr['candidate_recall']:.4f}"
          f"  <- ceiling: the re-ranker can never beat this,")
    print( "                                    because it only reorders what "
           "stage 1 retrieved.")

    print("\n  Re-ranker feature importances:")
    for name, val in rr["importances"].items():
        print(f"    {name:<20} {val}")

    # ---------------- eyeball one user -------------------------------------
    u = int(test_users[0])
    top = np.argsort(-scores[u])[:5]
    seen = sorted(ds.train_items_by_user[u])[:5]
    print(f"\n  Sanity check - user {u}")
    print( "    watched (sample):")
    for i in seen:
        print(f"      - {ds.item_titles.get(i, '?')}")
    print( "    top-5 recommended:")
    for i in top:
        print(f"      - {ds.item_titles.get(int(i), '?')}")
    print(f"    held-out truth: {ds.item_titles.get(int(test_items[0]), '?')}")

    save_artifacts(ds, model, rr, tt_metrics, pop_metrics, nolq_metrics)
    print(f"\n  total runtime: {time.time() - t0:.1f}s\n")


def save_artifacts(ds, model, rr, tt_metrics, pop_metrics, nolq_metrics):
    """Persist everything the demo app needs so it never retrains at runtime.

    This is the "model versioning" half of the lifecycle: the app depends on a
    frozen artifact, not on a training run happening to produce the same
    numbers again.
    """
    import json, pickle, torch
    from pathlib import Path

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    from two_tower import history_matrix

    with torch.no_grad():
        genres = torch.from_numpy(ds.genres)
        item_vecs = model.all_item_vectors(genres)
        H = history_matrix(ds)
        user_means = (H @ item_vecs) / H.sum(1, keepdim=True).clamp(min=1)
        user_vecs = model.encode_user_from_mean(user_means).numpy()
        item_vecs = item_vecs.numpy()

        # The user tower's weights, so the BROWSER can encode a visitor's
        # picks. It is two Linear layers with a ReLU between -- small enough
        # to re-implement in a few lines of JavaScript, which is what makes
        # the live demo able to serve a user the model never trained on.
        sd = model.user_mlp.state_dict()
        user_mlp = {
            "w0": sd["0.weight"].numpy(), "b0": sd["0.bias"].numpy(),
            "w2": sd["2.weight"].numpy(), "b2": sd["2.bias"].numpy(),
        }

    np.savez_compressed(
        out / "embeddings.npz", item_vecs=item_vecs, user_vecs=user_vecs,
        genres=ds.genres, item_year=ds.item_year,
        item_mean_rating=ds.item_mean_rating, item_n_ratings=ds.item_n_ratings,
        **{f"user_mlp_{k}": v for k, v in user_mlp.items()},
    )
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({
            "item_titles": ds.item_titles,
            "train_items_by_user": ds.train_items_by_user,
            "n_users": ds.n_users, "n_items": ds.n_items,
            "train_item_counts": np.bincount(ds.train.item.to_numpy(),
                                             minlength=ds.n_items),
        }, f)

    metrics = {
        "popularity_baseline": pop_metrics,
        "two_tower_no_logq": nolq_metrics,
        "two_tower_logq": tt_metrics,
        "two_tower_logq_reranked": rr["reranked"],
        "candidate_recall@100": rr["candidate_recall"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  artifacts written to {out}/ "
          f"({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
