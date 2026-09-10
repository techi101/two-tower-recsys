# Two-Stage Recommender: Two-Tower Retrieval + LightGBM Re-ranking

**[▶ Live demo](https://techi101.github.io/two-tower-recsys/)** — runs entirely in your
browser, no server.

A production-shaped recommender built on MovieLens-100K, implementing the
**retrieve-then-rank** architecture used by large-scale recommendation systems
at YouTube, Spotify, Pinterest and most e-commerce platforms.

Stage 1 narrows 1,447 candidates to ~100 with a PyTorch two-tower model.
Stage 2 re-ranks those with a LightGBM LambdaMART ranker. Everything is
evaluated against a popularity baseline, because a recommender that cannot
beat *"show everyone the most popular thing"* has learned nothing about
individual users.

---

## Results

Chronological leave-one-out evaluation on 938 held-out interactions.
Already-seen items are masked; the full catalogue is ranked.

| Model | Recall@10 | Recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|
| 1. Popularity baseline | 0.0810 | 0.2324 | 0.0406 | 0.0391 |
| 2. Two-tower, no logQ correction | 0.0608 | 0.2612 | 0.0270 | 0.0291 |
| 3. Two-tower **+ logQ correction** | 0.1194 | **0.3657** | 0.0601 | 0.0578 |
| 4. **+ LightGBM re-ranking** | **0.1226** | 0.3486 | **0.0636** | **0.0582** |

- logQ correction (2 → 3): **+122.7% NDCG@10**
- Re-ranking (3 → 4): **+5.7% NDCG@10**
- Full pipeline vs. baseline: **+56.5% NDCG@10**, **+51.4% Recall@10**

Reproduce with `python run.py` (~100s on CPU, no GPU required).

---

## The bug that mattered

The first working version **lost to the popularity baseline** — NDCG@10 of
0.0270 against 0.0406 — and its recommendations were visibly wrong:

```
top-5 for user 0:  Jupiter's Wife (1994)
                   I Shot Andy Warhol (1996)
                   Underground (1995)
                   Basquiat (1996)
                   Heidi Fleiss: Hollywood Madam (1995)
```

Obscure arthouse films for a user whose history was *Toy Story*,
*Twelve Monkeys*, *Dead Man Walking*.

**Cause: sampling bias in in-batch negatives.** Negatives are drawn from
whatever else is in the batch, so an item appears as a negative in proportion
to its frequency in the data. A blockbuster is punished in nearly every batch;
an obscure film almost never is. The model learned to avoid popular items —
popularity bias, running backwards.

**Fix: logQ correction** (from Google's *Sampling-Bias-Corrected Neural
Modeling for Large Corpus Item Recommendations*). Subtract each item's log
sampling probability from its logit:

```
corrected_logit(i, j) = raw_logit(i, j) − log Q(j)
```

An item sampled 100× more often loses log(100) from its score, exactly
cancelling its over-sampling penalty. Applied at **training time only** —
at inference there is no sampling, so there is no bias to correct.

NDCG@10: 0.0270 → 0.0601. The same user now gets *A Clockwork Orange*,
*Trainspotting*, *Short Cuts*.

Both variants are kept in the code (`logq_correction=True|False`) and both are
reported above, so the ablation is reproducible rather than a claim.

---

## Architecture

### Stage 1 — Two-tower retrieval (PyTorch)

The user and the item are encoded **separately** into the same vector space,
so a score is only a dot product:

```
score(u, i) = user_vector(u) · item_vector(i)
```

That restriction is the entire point. Because the towers never interact until
the final dot product, **every item vector can be computed offline** and stored
in a vector index. Serving becomes one nearest-neighbour lookup, independent of
catalogue size. The cost is expressiveness: the model structurally cannot learn
"this user likes Horror *only when* it is also highly rated."

- **User tower:** id embedding → MLP → L2 normalise
- **Item tower:** id embedding **+ genre projection** → MLP → L2 normalise
  The genre branch is what gives a zero-interaction item a usable vector —
  the cold-start path.
- **Loss:** in-batch negatives / InfoNCE. For each user in a batch of 512, the
  other 511 users' items serve as negatives. False negatives (two users sharing
  a positive item) are masked out.
- **Temperature:** 0.07, because cosine similarities in [−1, 1] produce a
  near-uniform softmax otherwise.

### Stage 2 — LambdaMART re-ranking (LightGBM)

Re-scores only the ~100 survivors using **28 features** stage 1 cannot
represent — chiefly explicit user × item crosses:

| Feature group | Examples |
|---|---|
| Stage-1 signal | `two_tower_score`, `two_tower_rank` |
| Item quality | `item_mean_rating`, `item_popularity`, `item_n_ratings`, `item_year` |
| User state | `user_activity` |
| **User × item crosses** | `genre_affinity`, plus 19 per-genre affinity features |

`LGBMRanker(objective="lambdarank")` optimises the **order** of each user's
list rather than per-item probabilities: it weights each pairwise swap by how
much fixing it would improve NDCG, so getting position 1 right matters far more
than position 80. This is why LightGBM needs the `group` argument.

Top features by importance: `two_tower_rank`, `item_mean_rating`,
`genre_affinity`, `user_activity`.

---

## Avoiding leakage

Three decisions that keep the numbers honest — each is a way these results
could have been silently inflated:

1. **Chronological split, not random.** For each user the *last* interaction is
   test and the second-to-last is validation. A random split lets the model see
   a user's future and predict their past.
2. **Seen items masked at scoring time.** Recommending something already
   watched is not a win.
3. **Separate candidate sets for training and evaluating the re-ranker.**
   Training rows are labelled against the *validation* item, evaluation rows
   against the *test* item. `item_mean_rating` is computed with the held-out
   `(user, item)` pairs removed, so the label never leaks into a feature.

Model selection uses validation NDCG@10, never training loss — training loss
keeps falling long after ranking quality peaks.

---

## Honest limitations

- **Candidate recall@100 is 0.5213.** For nearly half of users the correct
  item never reaches stage 2 at all. That is a hard ceiling on the pipeline,
  and it means *improving retrieval would pay far more than further ranking
  work* — the re-ranker is already near the limit of what it was handed.
- **Recall@50 drops after re-ranking** (0.3657 → 0.3486). Expected, not a bug:
  the ranker optimises NDCG@10, pulling items toward the top and pushing others
  past position 50. Precision at the head is bought with depth.
- **MovieLens-100K is small** — 938 users after filtering. These numbers would
  not transfer unchanged to a production catalogue.
- **Brute-force search, not ANN.** With 1,447 items an exact dot product over
  the full matrix is faster than an index. At million-item scale this is where
  FAISS or ScaNN slots in — the embeddings are already the right shape for it.
- **No online A/B test.** All results are offline. Offline NDCG gains do not
  reliably predict engagement gains; a real deployment would need an
  interleaving or A/B experiment to confirm.

## What I would do next

1. Raise candidate recall — larger candidate set, or a hybrid of two-tower +
   item-item co-occurrence retrieval, since that ceiling dominates everything.
2. Sequence modelling for the user tower (GRU4Rec / SASRec) so recent watches
   count more than old ones; the current user embedding is order-blind.
3. Swap brute force for FAISS and measure p99 latency at scale.
4. An interleaving harness to compare rankers online.

---

## Running it

```bash
# 1. dependencies
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. data (not committed - MovieLens license restricts redistribution)
python scripts/download_data.py

# 3. train + evaluate + write artifacts   (~100s, CPU only)
python run.py

# 4. demo app
pip install -r requirements.txt
streamlit run app.py
```

### Deployment

The demo app depends on **numpy and streamlit only** — no torch. Training
writes frozen embeddings to `artifacts/` (0.8 MB, committed), and serving is a
matrix multiply against those. Separating the training stack from the serving
stack is why the app cold-starts in seconds instead of installing CUDA wheels.

**Live at [techi101.github.io/two-tower-recsys](https://techi101.github.io/two-tower-recsys/)**,
hosted as a static site with no backend at all. A two-tower model's inference
path is a dot product between the user vector and the item matrix — the towers
already did their work offline — so `docs/index.html` loads the exported
embeddings and computes recommendations in JavaScript. That is a property of
the architecture, not a trick: it is the same property that lets production
systems serve two-tower retrieval from a vector index.

The static demo runs **stage 1 only**; the re-ranker needs per-candidate
feature computation. For the full pipeline, `app.py` runs on
[Streamlit Community Cloud](https://share.streamlit.io) (point it at this repo,
entrypoint `app.py`), and a `Dockerfile` is included for any container host.

---

## Layout

```
src/data.py        loading, chronological split, leak-free item features
src/two_tower.py   two-tower model, logQ correction, metrics, training loop
src/rerank.py      candidate generation, feature engineering, LambdaMART
run.py             end-to-end pipeline + ablation + artifact export
app.py             Streamlit demo (serving only)
scripts/           dataset download with content validation
```

The source files are commented to connect the **theory to the code** —
why two towers, why in-batch negatives, why a temperature, why LambdaRank
instead of classification — rather than restating what each line does.

## Transferring this to another domain

Only `src/data.py` is MovieLens-specific. The architecture is domain-agnostic:

| Domain | user side | item side |
|---|---|---|
| E-commerce | shopper | products |
| Music / video | listener | tracks |
| Food delivery | customer | restaurants |
| Jobs | candidate | postings |
| Search / RAG | query | documents |

The last row is worth noting: a two-tower model **is** a bi-encoder, the same
architecture that embeds document chunks for retrieval-augmented generation,
and the re-ranker is the cross-encoder stage. Retrieve-then-rank is the same
pattern in both fields.
