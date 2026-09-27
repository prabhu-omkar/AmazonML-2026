"""Memory-lean loaders working on the normalised part files (ids as int64)."""
import glob

from common import *
from common import _part_no

FIELD_COLS = ["entity_id", "country", "nm", "core", "ad", "nums", "alph", "is_dom", "has_dba"]


def id_to_int(ids):
    return np.array([int(x[1]) * 10 ** 10 + int(x[3:]) for x in ids], dtype=np.int64)


def int_to_id(v):
    return np.array([f"S{x // 10 ** 10}-{x % 10 ** 10}" for x in np.asarray(v).tolist()], dtype=object)


def part_files(work, split, k):
    fs = glob.glob(f"{work}/norm_{split}_s{k}.p*.pkl") + glob.glob(f"{work}/norm_{split}_s{k}.p*.parquet")
    return sorted(fs, key=_part_no)


def read_part(path, cols):
    if path.endswith(".parquet"):
        d = pd.read_parquet(path, columns=cols)
    else:
        d = pd.read_pickle(path)[cols]
    for c in d.columns:
        if d[c].dtype != object and not pd.api.types.is_numeric_dtype(d[c].dtype):
            d[c] = d[c].astype(object)
    return d


def load_fields(work, split, ks, ids=None, country=None, cols=FIELD_COLS):
    """ids: optional np.int64 array / set of int ids to keep"""
    out = []
    idx = None if ids is None else pd.Index(np.unique(np.asarray(list(ids) if isinstance(ids, set) else ids, dtype=np.int64)))
    for k in ks:
        for f in part_files(work, split, k):
            d = read_part(f, cols)
            if country is not None:
                d = d[d.country == country]
            d = d.copy()
            d["entity_id"] = id_to_int(d.entity_id.values)
            if idx is not None:
                d = d[idx.get_indexer(d.entity_id.values) >= 0]
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def corpus_sample(work, split, country, n=400000, seed=0):
    """random sample of S2+S3 records of a country (for IDF fitting)"""
    parts = []
    for k in (2, 3):
        for f in part_files(work, split, k):
            d = read_part(f, FIELD_COLS)
            d = d[d.country == country]
            parts.append(d.sample(frac=min(1.0, 0.1), random_state=seed))
    d = pd.concat(parts, ignore_index=True)
    if len(d) > n:
        d = d.sample(n=n, random_state=seed)
    return d.reset_index(drop=True)


def gt_int(path):
    gt = read_ground_truth(path)
    out = {}
    for s, lst in gt.items():
        out[int(s[3:]) + 10 ** 10] = set(int(x[1]) * 10 ** 10 + int(x[3:]) for x in lst)
    return out
