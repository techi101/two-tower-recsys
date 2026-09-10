"""
Stage 1: TWO-TOWER RETRIEVAL MODEL (history-based user tower).

THEORY -> CODE NOTES
--------------------
WHY TWO TOWERS INSTEAD OF ONE MODEL?
    A single model taking (user, item) together and outputting a score is more
    expressive -- but to recommend you would run it once per candidate item.
    With a million items and a 100ms budget that is impossible.

    Two towers force the user and item to be encoded SEPARATELY into the same
    vector space, so the score is only a dot product:

        score(u, i) = user_vector(u) . item_vector(i)

    That constraint is what makes it fast: every item vector is computed
    OFFLINE and stored in a vector index, and serving becomes one
    nearest-neighbour lookup regardless of catalogue size.

WHY THE USER TOWER READS HISTORY, NOT A USER ID
    The obvious user tower is nn.Embedding(n_users, dim) -- one learned vector
    per user id. It trains well and scores well, and it is USELESS IN
    PRODUCTION, because it can only represent users who existed at training
    time. A new signup has no row in that table. Neither does anyone browsing
    logged-out. You cannot recommend anything to them.

    So this user tower takes the SET OF ITEMS the person liked and encodes
    that instead:

        user_vector(S) = normalise( user_mlp( mean_{s in S} item_vector(s) ) )

    Same output space, same dot-product scoring, but now the input is
    behaviour rather than identity. Anyone who has clicked on two things has a
    user vector. This is the standard "fold-in" formulation, and it is what
    lets the live demo accept a stranger's picks instead of only replaying
    pre-existing user ids.

    The cost: the mean is ORDER-BLIND. It cannot tell that you watched horror
    last week and comedies last year. A sequence model (GRU4Rec, SASRec) is
    the upgrade path, and is listed in the README as future work.

LEAKAGE TRAP -- the part that is easy to get wrong
    When training on the pair (user u, positive item i), the user vector must
    be built from u's history WITHOUT i. Include it and the model learns the
    identity function: "recommend the item that is already in the input."
    Validation NDCG would look spectacular and the model would be worthless.

    Below, user_sums holds the sum of each user's item vectors; for a training
    pair we subtract the target's vector and divide by (count - 1). O(1) per
    example, and exactly leakage-free.

WHY IN-BATCH NEGATIVES?
    Our only signal is positives. A model trained on positives alone collapses
    -- it can score everything high and never be wrong. For user i in a batch,
    the positive item of every OTHER user j acts as a negative, giving 511
    free negatives at batch size 512. The (B x B) score matrix has the correct
    answers on its diagonal, so the loss is plain cross-entropy per row
    (InfoNCE / sampled softmax).

    False negatives -- two users in a batch sharing a positive item -- are
    masked out below.

WHY A TEMPERATURE?
    Both towers are L2-normalised, so the dot product is a cosine in [-1, 1].
    Cross-entropy over logits that small is nearly uniform and learns slowly.
    Dividing by 0.07 sharpens the softmax. Standard contrastive-learning trick.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """
    LOGQ CORRECTION -- the fix for in-batch sampling bias
    -----------------------------------------------------
    In-batch negatives are not sampled uniformly: an item appears in a batch in
    proportion to its frequency, so a blockbuster is a "negative" in nearly
    every batch while an obscure film almost never is. The model is punished
    far more for scoring popular items highly and learns to avoid them --
    popularity bias, running backwards. Uncorrected, this model recommended
    obscure arthouse titles to everyone and lost to a popularity baseline.

    The correction (Google, "Sampling-Bias-Corrected Neural Modeling for Large
    Corpus Item Recommendations") subtracts each item's log sampling
    probability from its logit:

        corrected_logit(i, j) = raw_logit(i, j) - log Q(j)

    An item sampled 100x more often loses log(100), exactly cancelling its
    over-sampling penalty. TRAINING ONLY -- at inference nothing is being
    sampled, so there is no bias to correct.
    """

    def __init__(self, n_items, n_genres, emb_dim=64, hidden=128,
                 item_log_q=None):
        super().__init__()

        # --- ITEM TOWER -----------------------------------------------------
        # id embedding (collaborative signal: "people who liked X liked Y")
        # + genre projection (content signal, and the part that still works
        # for an item with zero interactions -- the item cold-start path).
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.genre_proj = nn.Linear(n_genres, emb_dim, bias=False)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

        # --- USER TOWER -----------------------------------------------------
        # Consumes the MEAN OF ITEM VECTORS, not a user id. Deliberately small
        # and exportable: two Linear layers with a ReLU is ~16k parameters,
        # which the browser demo re-implements in a few lines of JavaScript so
        # a visitor's picks can be encoded client-side.
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

        self.temperature = 0.07

        if item_log_q is None:
            self.register_buffer("item_log_q", None)
        else:
            self.register_buffer("item_log_q",
                                 torch.as_tensor(item_log_q).float())

        nn.init.normal_(self.item_emb.weight, std=0.05)

    # -- item side ----------------------------------------------------------
    def encode_item(self, item_ids, genres):
        v = self.item_emb(item_ids) + self.genre_proj(genres)
        return F.normalize(self.item_mlp(v), dim=-1)

    def all_item_vectors(self, genres):
        ids = torch.arange(genres.shape[0], device=genres.device)
        return self.encode_item(ids, genres)

    # -- user side ----------------------------------------------------------
    def encode_user_from_mean(self, mean_item_vec):
        """mean_item_vec: (B, d) average of the item vectors the user liked."""
        return F.normalize(self.user_mlp(mean_item_vec), dim=-1)

    def encode_user_from_items(self, item_vecs, item_lists):
        """Encode users from explicit lists of liked item indices (inference).

        This is the path the live demo uses: someone picks a handful of movies
        and gets a user vector, with no training-time identity required.
        """
        means = torch.stack([item_vecs[torch.as_tensor(lst)].mean(0)
                             for lst in item_lists])
        return self.encode_user_from_mean(means)

    # -- training -----------------------------------------------------------
    def forward(self, user_mean_vecs, item_ids, item_vecs_batch):
        u = self.encode_user_from_mean(user_mean_vecs)   # (B, d)
        i = item_vecs_batch                              # (B, d), already unit

        logits = (u @ i.T) / self.temperature

        if self.item_log_q is not None:
            logits = logits - self.item_log_q[item_ids][None, :]

        same_item = item_ids[:, None] == item_ids[None, :]
        eye = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_item & ~eye, float("-inf"))

        labels = torch.arange(len(item_ids), device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
# Exactly one held-out item per user, so:
#   Recall@K : 1 if it lands in the top K. "Did we surface it at all" -- the
#              metric that matters for RETRIEVAL, whose job is not to lose the
#              right answer.
#   NDCG@K   : 1/log2(rank+2) within top K. Pays more for putting it near the
#              TOP -- the metric that matters for RANKING and tracks what a
#              user actually experiences.
#   MRR      : 1/(rank+1), no cutoff.
# ---------------------------------------------------------------------------

def ranking_metrics(ranks, ks=(10, 50)):
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {f"Recall@{k}": float((ranks < k).mean()) for k in ks}
    out["NDCG@10"] = float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).mean())
    out["MRR"] = float((1.0 / (ranks + 1)).mean())
    return out


def history_matrix(ds, device="cpu"):
    """Dense (n_users, n_items) binary matrix of TRAIN interactions.

    Dense is fine here: 938 x 1447 floats is 5.4 MB. At real scale this is a
    sparse matmul or an embedding-bag, not a dense matrix.
    """
    H = torch.zeros(ds.n_users, ds.n_items, device=device)
    H[torch.from_numpy(ds.train.user.to_numpy().copy()).long(),
      torch.from_numpy(ds.train.item.to_numpy().copy()).long()] = 1.0
    return H


@torch.no_grad()
def score_all_items(model, ds, mask_val=True, device="cpu"):
    """(n_users, n_items) scores with already-seen items masked to -inf.

    Masking matters: recommending something the user already watched is not a
    win, and leaving it unmasked silently inflates every metric.
    """
    model.eval()
    genres = torch.from_numpy(ds.genres).to(device)
    item_vecs = model.all_item_vectors(genres)

    H = history_matrix(ds, device)
    counts = H.sum(1, keepdim=True).clamp(min=1)
    user_means = (H @ item_vecs) / counts
    user_vecs = model.encode_user_from_mean(user_means)

    scores = (user_vecs @ item_vecs.T).cpu().numpy()

    for u, seen in ds.train_items_by_user.items():
        scores[u, list(seen)] = -np.inf
    if mask_val:
        for u, i in zip(ds.val.user.to_numpy(), ds.val.item.to_numpy()):
            scores[u, i] = -np.inf
    return scores


def rank_of_targets(scores, users, targets):
    return np.array([int((scores[u] > scores[u][t]).sum())
                     for u, t in zip(users, targets)])


def popularity_baseline(ds, users, targets, mask_val=True):
    """The baseline every recommender must beat: rank by global popularity.

    A model that cannot beat "show everyone the most watched thing" has
    learned nothing about individuals. Reporting metrics without this
    comparison is the most common way recsys results mislead.
    """
    counts = np.bincount(ds.train.item.to_numpy(),
                         minlength=ds.n_items).astype(np.float64)
    scores = np.tile(counts, (ds.n_users, 1))
    for u, seen in ds.train_items_by_user.items():
        scores[u, list(seen)] = -np.inf
    if mask_val:
        for u, i in zip(ds.val.user.to_numpy(), ds.val.item.to_numpy()):
            scores[u, i] = -np.inf
    return ranking_metrics(rank_of_targets(scores, users, targets))


def item_log_q(ds):
    """log P(item is sampled), estimated from train-set frequency."""
    counts = np.bincount(ds.train.item.to_numpy(), minlength=ds.n_items)
    probs = (counts + 1) / (counts.sum() + ds.n_items)
    return np.log(probs).astype(np.float32)


def train_two_tower(ds, epochs=30, batch_size=512, lr=1e-3, seed=0,
                    device="cpu", verbose=True, logq_correction=True,
                    deterministic=True):
    """
    DETERMINISM NOTE
    ----------------
    The user tower sums each user's item vectors with a matmul every step.
    Floating-point addition is not associative, so a multithreaded reduction
    sums in a different order each run and the results drift -- measured at
    +/-5% NDCG@10 across identically-seeded runs on 10 threads.

    Seeding alone does NOT fix that; it is a thread-scheduling effect, not an
    RNG one. Pinning to one thread makes the reduction order fixed and the
    run bit-identical, at roughly 2x the wall time. Left on by default,
    because a benchmark you cannot reproduce is not a benchmark.
    """
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_q = item_log_q(ds) if logq_correction else None
    model = TwoTower(ds.n_items, ds.genres.shape[1], item_log_q=log_q).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    users = torch.from_numpy(ds.train.user.to_numpy().copy()).long().to(device)
    items = torch.from_numpy(ds.train.item.to_numpy().copy()).long().to(device)
    genres = torch.from_numpy(ds.genres).to(device)
    H = history_matrix(ds, device)
    counts = H.sum(1)                     # interactions per user
    n = len(users)

    val_users, val_items = ds.val.user.to_numpy(), ds.val.item.to_numpy()
    best_ndcg, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 8:
                continue
            b_users, b_items = users[idx], items[idx]

            # Item vectors for the whole catalogue, recomputed each step so
            # gradients flow into the item tower through the user side too.
            item_vecs = model.all_item_vectors(genres)

            # LEAKAGE-FREE user means: sum of the user's item vectors MINUS
            # the target item, divided by (count - 1). Without the subtraction
            # the target sits inside its own query and the task is trivial.
            # Only the batch's rows of H are needed, not all n_users.
            b_sums = (H[b_users] @ item_vecs) - item_vecs[b_items]
            b_counts = (counts[b_users] - 1.0).clamp(min=1).unsqueeze(1)
            b_means = b_sums / b_counts

            loss = model(b_means, b_items, item_vecs[b_items])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        # Model selection on VALIDATION NDCG, never training loss -- training
        # loss keeps falling long after ranking quality peaks, and that gap is
        # exactly the overfitting worth detecting.
        if epoch % 3 == 0 or epoch == epochs:
            s = score_all_items(model, ds, mask_val=False, device=device)
            m = ranking_metrics(rank_of_targets(s, val_users, val_items))
            if verbose:
                print(f"  epoch {epoch:3d} | loss {total/n:.4f} "
                      f"| val NDCG@10 {m['NDCG@10']:.4f} "
                      f"| val Recall@50 {m['Recall@50']:.4f}")
            if m["NDCG@10"] > best_ndcg:
                best_ndcg = m["NDCG@10"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if verbose:
        print(f"  restored best checkpoint (val NDCG@10 {best_ndcg:.4f})")
    return model
