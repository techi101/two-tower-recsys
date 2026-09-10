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
    user_vecs = z["user_vecs"].astype(np.float32)

    # Raw little-endian float32, read straight into a Float32Array in JS.
    (OUT / "item_vecs.bin").write_bytes(item_vecs.tobytes())
    (OUT / "user_vecs.bin").write_bytes(user_vecs.tobytes())

    titles = meta["item_titles"]
    history = meta["train_items_by_user"]

    payload = {
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "dim": int(item_vecs.shape[1]),
        # index-aligned so JS can use position, not a string key lookup
        "titles": [titles.get(i, f"item {i}") for i in range(meta["n_items"])],
        "history": [sorted(int(x) for x in history.get(u, []))
                    for u in range(meta["n_users"])],
        "metrics": metrics,
    }
    (OUT / "meta.json").write_text(json.dumps(payload, separators=(",", ":")))

    total = sum(f.stat().st_size for f in OUT.iterdir())
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<16} {f.stat().st_size / 1024:>8.1f} KB")
    print(f"  {'total':<16} {total / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
