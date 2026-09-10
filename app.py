"""
Streamlit demo for the two-stage recommender.

Loads FROZEN ARTIFACTS produced by `run.py` -- it never trains at request time.
That separation is the point: training is a batch job, serving is a lookup.

    streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

ART = Path("artifacts")

st.set_page_config(page_title="Two-Stage Recommender", page_icon="🎬",
                   layout="wide")


@st.cache_resource
def load_artifacts():
    z = np.load(ART / "embeddings.npz")
    with open(ART / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    metrics = json.loads((ART / "metrics.json").read_text())
    return z, meta, metrics


if not (ART / "embeddings.npz").exists():
    st.error("No artifacts found. Run `python run.py` first to train the model.")
    st.stop()

z, meta, metrics = load_artifacts()
item_vecs = z["item_vecs"]          # (n_items, d), L2-normalised
user_vecs = z["user_vecs"]          # (n_users, d), L2-normalised
titles = meta["item_titles"]
history = meta["train_items_by_user"]
n_users, n_items = meta["n_users"], meta["n_items"]

st.title("🎬 Two-Stage Recommender")
st.caption(
    "Two-tower retrieval (PyTorch) → LightGBM LambdaMART re-ranking, "
    "trained on MovieLens-100K. The same retrieve-then-rank architecture "
    "used for product, music, job and feed recommendations."
)

# ---------------------------------------------------------------- metrics --
m = metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("NDCG@10", f"{m['two_tower_logq_reranked']['NDCG@10']:.4f}",
          f"{(m['two_tower_logq_reranked']['NDCG@10'] / m['popularity_baseline']['NDCG@10'] - 1) * 100:+.0f}% vs baseline")
c2.metric("Recall@10", f"{m['two_tower_logq_reranked']['Recall@10']:.4f}",
          f"{(m['two_tower_logq_reranked']['Recall@10'] / m['popularity_baseline']['Recall@10'] - 1) * 100:+.0f}% vs baseline")
c3.metric("Recall@50", f"{m['two_tower_logq']['Recall@50']:.4f}",
          f"{(m['two_tower_logq']['Recall@50'] / m['popularity_baseline']['Recall@50'] - 1) * 100:+.0f}% vs baseline")
c4.metric("logQ correction lift",
          f"{(m['two_tower_logq']['NDCG@10'] / m['two_tower_no_logq']['NDCG@10'] - 1) * 100:+.0f}%",
          "NDCG@10")

tab1, tab2, tab3 = st.tabs(
    ["Recommend for a user", "Similar movies", "How it works"])

# ------------------------------------------------------- tab 1: recommend --
with tab1:
    uid = st.selectbox("User", range(n_users), index=0,
                       format_func=lambda u: f"user {u}  "
                                             f"({len(history.get(u, [])):>3} movies watched)")

    seen = history.get(uid, set())
    scores = item_vecs @ user_vecs[uid]           # the entire retrieval step
    scores[list(seen)] = -np.inf                  # never re-recommend

    topk = np.argsort(-scores)[:10]

    left, right = st.columns(2)
    with left:
        st.subheader("Watched")
        for i in sorted(seen)[:15]:
            st.write(f"• {titles.get(i, '?')}")
        if len(seen) > 15:
            st.caption(f"…and {len(seen) - 15} more")
    with right:
        st.subheader("Recommended")
        for rank, i in enumerate(topk, 1):
            st.write(f"**{rank}.** {titles.get(int(i), '?')}  "
                     f"`{scores[i]:.3f}`")

# --------------------------------------------------- tab 2: similar items --
with tab2:
    st.write(
        "Nearest neighbours in the learned item embedding space. Nothing here "
        "uses genres directly — similarity emerges from *who watched what*, "
        "which is what makes it collaborative filtering rather than "
        "content matching."
    )
    valid = sorted(titles.items(), key=lambda kv: kv[1])
    choice = st.selectbox("Movie", [k for k, _ in valid],
                          format_func=lambda i: titles[i])

    sims = item_vecs @ item_vecs[choice]
    sims[choice] = -np.inf
    for rank, i in enumerate(np.argsort(-sims)[:10], 1):
        st.write(f"**{rank}.** {titles.get(int(i), '?')}  "
                 f"`cos = {sims[i]:.3f}`")

# --------------------------------------------------------- tab 3: explain --
with tab3:
    st.markdown(
        """
### The two stages

**Stage 1 — retrieval (two-tower neural network).**
The user and the movie are encoded *separately* into the same vector space,
so a score is just a dot product. That restriction is what makes it fast:
every item vector is computed offline and stored in a vector index, so
serving is one nearest-neighbour lookup regardless of catalogue size.
Trained with **in-batch negatives** — for each user in a batch, the other
users' items act as negatives.

**Stage 2 — ranking (LightGBM LambdaMART).**
Re-scores only the ~100 survivors using features stage 1 structurally cannot
represent, such as per-genre affinity crosses between this user and this
candidate. LambdaRank optimises the *order* of the list rather than
per-item probabilities.

### The bug that mattered

The first version **lost to a popularity baseline** (NDCG@10 0.0270 vs 0.0406)
and recommended obscure arthouse films. Cause: in-batch negatives sample items
in proportion to popularity, so blockbusters appear as negatives constantly and
the model learns to avoid them — popularity bias, inverted.

Fix: **logQ correction** — subtract each item's log sampling probability from
its logit, cancelling the over-sampling penalty. NDCG@10 went 0.0270 → 0.0601,
**+123%**.

### Honest caveats

- **Recall@50 drops slightly after re-ranking** (0.3657 → 0.3486). Expected:
  the re-ranker optimises NDCG@10, so it pulls items toward the top and pushes
  others past position 50. Precision at the top is bought with depth.
- **Candidate recall@100 is 0.52** — the true item never even reaches stage 2
  for half of users. That is the pipeline's ceiling, and improving retrieval
  would pay far more than further ranking work.
- MovieLens-100K is small (938 users). These numbers would not transfer
  unchanged to a production catalogue.
        """
    )
    st.json(metrics)
