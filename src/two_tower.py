"""
Stage 1 of the pipeline: TWO-TOWER RETRIEVAL MODEL.

THEORY -> CODE NOTES
--------------------
WHY TWO TOWERS INSTEAD OF ONE MODEL?
    A single model that takes (user, item) together and outputs a score is more
    expressive -- but to recommend, you would have to run it once per candidate
    item. With a million items and a 100ms latency budget, that is impossible.

    The two-tower architecture forces the user and the item to be encoded
    SEPARATELY into the same vector space, so the score is only a dot product:

        score(u, i) = user_vector(u) . item_vector(i)

    That constraint buys something huge: every item vector can be computed
    OFFLINE and stored in a vector index. At request time you encode the user
    once and do a nearest-neighbour search. Retrieval over millions of items
    becomes one ANN lookup.

    This is exactly the trade the job description means by "user embeddings,
    semantic retrieval": give up expressiveness at stage 1 to get speed, then
    buy the expressiveness back with a re-ranker at stage 2.

WHY IN-BATCH NEGATIVES?
    Our only training signal is positives ("this user watched this"). A model
    trained on positives alone collapses -- it can score everything high and
    never be wrong. We need negatives: items the user did not engage with.

    Instead of sampling negatives explicitly, we exploit the batch. For user i,
    the positive item of every OTHER user j in the batch acts as a negative.
    A batch of 512 gives 511 free negatives per user.

    Concretely we compute the (B x B) matrix of all user-item scores in the
    batch. The correct answer for row i is column i -- the diagonal. So the
    loss is plain cross-entropy over each row. This is InfoNCE / sampled
    softmax, the same objective CLIP and most modern retrieval models use.

    One subtlety handled below: if two users in a batch share the same positive
    item, that item appears as a "negative" for a user it is genuinely positive
    for -- a FALSE negative that pushes the model the wrong way. We mask those.

WHY A TEMPERATURE?
    We L2-normalise both towers, so the dot product is a cosine in [-1, 1].
    Cross-entropy over logits that small is nearly uniform and learns slowly.
    Dividing by a temperature (0.07) spreads the logits out and sharpens the
    softmax. Standard trick from contrastive learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """
    LOGQ CORRECTION -- the fix for in-batch sampling bias
    -----------------------------------------------------
    In-batch negatives are not sampled uniformly. An item appears in a batch in
    proportion to how often it occurs in the data, so a blockbuster shows up as
    a "negative" in nearly every batch while an obscure film almost never does.
    The model is therefore punished far more for scoring popular items highly,
    and it learns to avoid them -- popularity bias, running backwards. You can
    see it directly in the recommendations: the uncorrected model surfaces
    obscure arthouse titles nobody asked for.

    The correction (from Google's "Sampling-Bias-Corrected Neural Modeling for
    Large Corpus Item Recommendations", the YouTube two-tower paper) is to
    subtract each item's log sampling probability from its logit:

        corrected_logit(i, j) = raw_logit(i, j) - log Q(j)

    An item sampled 100x more often gets its score knocked down by log(100),
    which exactly cancels the advantage it got from being over-sampled. What
    remains is the true affinity signal.

    Note this applies during TRAINING ONLY. At inference we want the raw dot
    product -- there is no sampling happening, so there is no bias to correct.
    """

    def __init__(self, n_users, n_items, n_genres, emb_dim=64, hidden=128,
                 item_log_q=None):
        super().__init__()

        # --- USER TOWER -----------------------------------------------------
        # Pure id embedding: one learned vector per user, shaped entirely by
        # which items they interacted with during training.
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

        # --- ITEM TOWER -----------------------------------------------------
        # Two inputs summed: a learned id embedding (collaborative signal --
        # "people who liked X also liked Y") plus a projection of the genre
        # multi-hot (content signal, and the part that still works for an item
        # with zero interactions -- the cold-start story).
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.genre_proj = nn.Linear(n_genres, emb_dim, bias=False)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

        self.temperature = 0.07

        # log Q(j): log sampling probability of each item. Registered as a
        # buffer so it moves with the model but is not a learned parameter.
        # None => correction disabled, for the ablation.
        if item_log_q is None:
            self.register_buffer("item_log_q", None)
        else:
            self.register_buffer("item_log_q", torch.as_tensor(item_log_q).float())

        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.05)

    def encode_user(self, user_ids):
        v = self.user_mlp(self.user_emb(user_ids))
        return F.normalize(v, dim=-1)              # unit length -> dot == cosine

    def encode_item(self, item_ids, genres):
        v = self.item_emb(item_ids) + self.genre_proj(genres)
        v = self.item_mlp(v)
        return F.normalize(v, dim=-1)

    def forward(self, user_ids, item_ids, genres):
        u = self.encode_user(user_ids)             # (B, d)
        i = self.encode_item(item_ids, genres)     # (B, d)

        # (B, B): every user in the batch scored against every item in the batch
        logits = (u @ i.T) / self.temperature

        # logQ correction: penalise each column by how often that item gets
        # sampled, cancelling the popularity advantage of in-batch negatives.
        if self.item_log_q is not None:
            logits = logits - self.item_log_q[item_ids][None, :]

        # Mask false negatives -- the same item appearing for two users.
        same_item = item_ids[:, None] == item_ids[None, :]
        eye = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_item & ~eye, float("-inf"))

        # The correct column for row i is i: the diagonal.
        labels = torch.arange(len(item_ids), device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
# We hold out exactly ONE item per user, which simplifies the metrics a lot:
#
#   Recall@K : 1 if the held-out item lands in the top K, else 0. Averaged over
#              users, this is "how often did we surface the right thing at all".
#              This is the metric that matters for the RETRIEVAL stage -- its
#              job is to not lose the right answer, not to rank it first.
#
#   NDCG@K   : 1/log2(rank+2) if within the top K, else 0. Same idea as recall
#              but pays more for putting the right item near the TOP. This is
#              the metric that matters for the RANKING stage, and the one that
#              tracks what a user actually experiences.
#
#   MRR      : 1/(rank+1), no cutoff. Sensitive to the whole ranking.
# ---------------------------------------------------------------------------

def ranking_metrics(ranks, ks=(10, 50)):
    """ranks: array of 0-based positions of each user's held-out item."""
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float((ranks < k).mean())
    out["NDCG@10"] = float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).mean())
    out["MRR"] = float((1.0 / (ranks + 1)).mean())
    return out


@torch.no_grad()
def score_all_items(model, ds, mask_val=True, device="cpu"):
    """Full (n_users, n_items) score matrix with ALREADY-SEEN ITEMS MASKED.

    Masking matters: recommending a movie the user already watched in training
    is not a win, and leaving it unmasked silently inflates every metric.
    Anything seen in train (and optionally val) is set to -inf.
    """
    model.eval()
    genres = torch.from_numpy(ds.genres).to(device)
    all_items = torch.arange(ds.n_items, device=device)
    all_users = torch.arange(ds.n_users, device=device)

    item_vecs = model.encode_item(all_items, genres)     # (n_items, d)
    user_vecs = model.encode_user(all_users)             # (n_users, d)
    scores = (user_vecs @ item_vecs.T).cpu().numpy()     # (n_users, n_items)

    for u, seen in ds.train_items_by_user.items():
        scores[u, list(seen)] = -np.inf
    if mask_val:
        for u, i in zip(ds.val.user.to_numpy(), ds.val.item.to_numpy()):
            scores[u, i] = -np.inf
    return scores


def rank_of_targets(scores, users, targets):
    """0-based rank of each user's held-out item inside their own score row."""
    ranks = []
    for u, t in zip(users, targets):
        row = scores[u]
        ranks.append(int((row > row[t]).sum()))   # how many items outscored it
    return np.array(ranks)


def popularity_baseline(ds, users, targets, mask_val=True):
    """The baseline every recommender must beat: just rank by global popularity.

    If a model cannot beat 'recommend whatever is most watched overall', it has
    learned nothing about the individual user. Reporting a metric without this
    comparison is the most common way recsys results mislead.
    """
    counts = np.bincount(ds.train.item.to_numpy(), minlength=ds.n_items).astype(np.float64)
    scores = np.tile(counts, (ds.n_users, 1))
    for u, seen in ds.train_items_by_user.items():
        scores[u, list(seen)] = -np.inf
    if mask_val:
        for u, i in zip(ds.val.user.to_numpy(), ds.val.item.to_numpy()):
            scores[u, i] = -np.inf
    return ranking_metrics(rank_of_targets(scores, users, targets))


def item_log_q(ds):
    """log P(item is sampled), estimated from its frequency in the train set."""
    counts = np.bincount(ds.train.item.to_numpy(), minlength=ds.n_items)
    probs = (counts + 1) / (counts.sum() + ds.n_items)   # +1 smoothing
    return np.log(probs).astype(np.float32)


def train_two_tower(ds, epochs=30, batch_size=512, lr=1e-3, seed=0,
                    device="cpu", verbose=True, logq_correction=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_q = item_log_q(ds) if logq_correction else None
    model = TwoTower(ds.n_users, ds.n_items, ds.genres.shape[1],
                     item_log_q=log_q).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    users = torch.from_numpy(ds.train.user.to_numpy().copy()).long().to(device)
    items = torch.from_numpy(ds.train.item.to_numpy().copy()).long().to(device)
    genre_tensor = torch.from_numpy(ds.genres).to(device)
    n = len(users)

    val_users, val_items = ds.val.user.to_numpy(), ds.val.item.to_numpy()
    best_ndcg, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 8:          # a tiny trailing batch gives too few negatives
                continue
            b_users, b_items = users[idx], items[idx]
            loss = model(b_users, b_items, genre_tensor[b_items])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        # Model selection on VALIDATION NDCG, never on training loss. Training
        # loss keeps falling long after ranking quality peaks -- that gap is
        # exactly the overfitting the job description asks about, and this is
        # how you detect it without a test-set peek.
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
