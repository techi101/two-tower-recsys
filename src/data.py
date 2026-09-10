"""
Data loading and train/val/test splitting for MovieLens-100K.

THEORY -> CODE NOTES
--------------------
1. Implicit feedback.
   Production recommenders rarely have clean star ratings; they have "did the
   user watch this / click this". We simulate that by treating rating >= 4 as a
   positive interaction and throwing everything else away. So the model never
   learns "predict the rating" -- it learns "which items belong with this user".

2. Leave-one-out split, ordered by TIME.
   For each user we sort their positives chronologically and hold out:
       last item        -> test
       second-to-last   -> validation
       everything else  -> train
   Splitting by time (not randomly) matters: a random split lets the model see
   a user's future and predict their past, which inflates every metric. This is
   the single most common way academic recsys numbers get faked by accident.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

POSITIVE_THRESHOLD = 4   # rating >= 4 counts as a positive interaction
MIN_POSITIVES = 5        # users need enough history to split train/val/test

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    train: pd.DataFrame          # columns: user, item  (contiguous 0-based ids)
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    genres: np.ndarray           # (n_items, 19) multi-hot genre matrix
    item_titles: dict            # new item id -> title, for eyeballing results
    train_items_by_user: dict    # user -> set(items), used to mask seen items
    item_year: np.ndarray        # (n_items,) release year, 0 if unknown
    item_mean_rating: np.ndarray # (n_items,) avg star rating, leak-free
    item_n_ratings: np.ndarray   # (n_items,) how many people rated it at all


def build_dataset(root="data/ml-100k"):
    ratings = pd.read_csv(
        f"{root}/u.data", sep="\t", names=["user", "item", "rating", "ts"]
    )

    # --- 1. implicit feedback: keep only positives -------------------------
    pos = ratings[ratings.rating >= POSITIVE_THRESHOLD].copy()

    # --- 2. drop users too short to split ----------------------------------
    counts = pos.groupby("user").size()
    pos = pos[pos.user.isin(counts[counts >= MIN_POSITIVES].index)]

    # --- 3. reindex ids to contiguous 0..N-1 -------------------------------
    # Embedding layers are lookup tables indexed by integer, so ids must be
    # dense. Raw MovieLens ids have gaps.
    raw_users = np.sort(pos.user.unique())
    raw_items = np.sort(pos.item.unique())
    u_map = {u: i for i, u in enumerate(raw_users)}
    i_map = {m: i for i, m in enumerate(raw_items)}
    pos["user"] = pos.user.map(u_map)
    pos["item"] = pos.item.map(i_map)

    # --- 4. chronological leave-one-out ------------------------------------
    pos = pos.sort_values(["user", "ts"])
    # cumcount(ascending=False): 0 == most recent interaction for that user
    pos["from_end"] = pos.groupby("user").cumcount(ascending=False)
    test = pos[pos.from_end == 0][["user", "item"]].reset_index(drop=True)
    val = pos[pos.from_end == 1][["user", "item"]].reset_index(drop=True)
    train = pos[pos.from_end >= 2][["user", "item"]].reset_index(drop=True)

    # --- 5. item side features (genres) ------------------------------------
    # These feed the ITEM TOWER. A pure-id embedding can only represent movies
    # it saw during training; genre features are what let a brand-new movie get
    # a sensible embedding on day one (the cold-start story).
    item_cols = ["item", "title", "release", "video_release", "imdb"] + GENRE_NAMES
    items = pd.read_csv(
        f"{root}/u.item", sep="|", names=item_cols, encoding="latin-1"
    )
    items = items[items.item.isin(i_map)].copy()
    items["item"] = items.item.map(i_map)
    items = items.sort_values("item")

    genres = items[GENRE_NAMES].to_numpy(dtype=np.float32)
    item_titles = dict(zip(items.item, items.title))

    # release year, parsed out of strings like "01-Jan-1995"
    item_year = np.zeros(len(raw_items), dtype=np.float32)
    years = pd.to_numeric(
        items.release.astype(str).str[-4:], errors="coerce").fillna(0)
    item_year[items.item.to_numpy()] = years.to_numpy(dtype=np.float32)

    # --- item quality signal, computed WITHOUT leakage ---------------------
    # Average star rating is a strong ranking feature, but it must not be
    # computed over the val/test interactions -- those are the answers. We
    # remove exactly those (user, item) pairs first, then aggregate.
    held_out = set(map(tuple, val[["user", "item"]].to_numpy())) | \
               set(map(tuple, test[["user", "item"]].to_numpy()))
    allr = ratings[ratings.user.isin(u_map) & ratings.item.isin(i_map)].copy()
    allr["user"] = allr.user.map(u_map)
    allr["item"] = allr.item.map(i_map)
    mask = [t not in held_out for t in zip(allr.user, allr.item)]
    allr = allr[mask]

    agg = allr.groupby("item").rating.agg(["mean", "count"])
    item_mean_rating = np.zeros(len(raw_items), dtype=np.float32)
    item_n_ratings = np.zeros(len(raw_items), dtype=np.float32)
    item_mean_rating[agg.index.to_numpy()] = agg["mean"].to_numpy()
    item_n_ratings[agg.index.to_numpy()] = agg["count"].to_numpy()

    train_items_by_user = train.groupby("user").item.apply(set).to_dict()

    return Dataset(
        train=train, val=val, test=test,
        n_users=len(raw_users), n_items=len(raw_items),
        genres=genres, item_titles=item_titles,
        train_items_by_user=train_items_by_user,
        item_year=item_year, item_mean_rating=item_mean_rating,
        item_n_ratings=item_n_ratings,
    )


if __name__ == "__main__":
    ds = build_dataset()
    print(f"users            : {ds.n_users}")
    print(f"items            : {ds.n_items}")
    print(f"train interactions: {len(ds.train)}")
    print(f"val interactions  : {len(ds.val)}")
    print(f"test interactions : {len(ds.test)}")
    print(f"genre matrix      : {ds.genres.shape}")
