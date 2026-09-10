"""
Export the trained artifacts into a form a browser can load, for the static
GitHub Pages demo in docs/.

Why this is possible at all: a two-tower model's entire inference path is a
dot product between a user vector and the item matrix. There is no neural
network to evaluate at request time -- the towers already did their work
offline. So "serving the model" is a matrix multiply, which JavaScript does
perfectly well. No Python, no server, no container.

That is a property of the ARCHITECTURE, not a trick: it is the same property
that lets production systems serve two-tower retrieval from a vector index.

    python scripts/export_web.py
"""

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(ART / "embeddings.npz")
    with open(ART / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    metrics = json.loads((ART / "metrics.json").read_text())

    item_vecs = z["item_vecs"].astype(np.float32)

    # Raw little-endian float32, read straight into a Float32Array in JS.
    (OUT / "item_vecs.bin").write_bytes(item_vecs.tobytes())

    titles = meta["item_titles"]
    counts = np.asarray(meta["train_item_counts"], dtype=int)

    GENRE_NAMES = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    genres = z["genres"]
    genre_lists = [[GENRE_NAMES[g] for g in np.nonzero(row)[0]
                    if GENRE_NAMES[g] != "unknown"] for row in genres]

    payload = {
        "n_items": int(meta["n_items"]),
        "dim": int(item_vecs.shape[1]),
        # index-aligned so JS uses position, not a string key lookup
        "titles": [titles.get(i, f"item {i}") for i in range(meta["n_items"])],
        "genres": genre_lists,
        "year": [int(y) for y in z["item_year"]],
        "popularity": counts.tolist(),
        # The user tower's weights. The browser runs this two-layer MLP on the
        # mean of whatever items a visitor picks, producing a user vector for
        # someone the model never trained on. Exporting it is what turns a
        # fixed set of demo users into a real recommender.
        "user_mlp": {
            "w0": z["user_mlp_w0"].tolist(), "b0": z["user_mlp_b0"].tolist(),
            "w2": z["user_mlp_w2"].tolist(), "b2": z["user_mlp_b2"].tolist(),
        },
        "metrics": metrics,
    }
    (OUT / "meta.json").write_text(json.dumps(payload, separators=(",", ":")))

    total = sum(f.stat().st_size for f in OUT.iterdir())
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<16} {f.stat().st_size / 1024:>8.1f} KB")
    print(f"  {'total':<16} {total / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
