"""Blend Kaggle cross-encoder scores with the GBDT (v6) scores; weight + threshold tuned on validation.

python src/blend_ce.py --work WORK --data DATA --ce_dir FOLDER_WITH_ce_val.parquet_AND_ce_test_*.parquet --out OUT
"""
import argparse
import glob
import os
import pickle

from common import *
from data import gt_int, int_to_id, load_fields
from train import postprocess, tune_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--art", default="", help="folder with models/gbdt_stage1.pkl, val_pairs.pkl, val_pred_stage1.npy, test_scored_stage1.pkl (default: --work)")
ap.add_argument("--data", required=True)
ap.add_argument("--ce_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=float, default=-1, help="force this GBDT weight (threshold still tuned on val)")
ap.add_argument("--tags", default="", help="force this CE subset, e.g. ce1,ce2,ce4,ce5 (default: best on validation)")
ap.add_argument("--blend_unseen", type=int, default=0, help="also blend CE into unseen countries (France)")
a = ap.parse_args()
if a.w < 0 and os.path.exists(f"{a.ce_dir}/blend_w.txt"):  # leaderboard-chosen weight (overrides val tuning)
    a.w = float(open(f"{a.ce_dir}/blend_w.txt").read().strip())
    log(f"using GBDT weight {a.w} from {a.ce_dir}/blend_w.txt")

gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
ART = a.art or a.work
b1 = pickle.load(open(f"{ART}/models/gbdt_stage1.pkl", "rb"))
Pv = pd.read_pickle(f"{ART}/val_pairs.pkl")
pv = np.load(f"{ART}/val_pred_stage1.npy").astype(np.float32)
key = lambda s, c: pd.MultiIndex.from_arrays([np.asarray(s, dtype=np.int64), np.asarray(c, dtype=np.int64)])


def ce_files(tag, kind):
    if tag == "ce1":
        pat = "ce_val*.parquet" if kind == "val" else "ce_test_*.parquet"
    else:
        pat = f"{tag}_val.parquet" if kind == "val" else f"{tag}_test_*.parquet"
    return sorted(glob.glob(f"{a.ce_dir}/{pat}"))


def aligned(tag, kind, s, c):
    d = pd.concat([pd.read_parquet(f, columns=["s1", "cand", "ce"]) for f in ce_files(tag, kind)], ignore_index=True)
    xf = f"{a.ce_dir}/{tag}_extra.parquet"  # extra pairs (v7-only candidates) scored in a later round
    if tag != "ce1" and os.path.exists(xf):
        e = pd.read_parquet(xf)
        d = pd.concat([d, e.loc[e.split == kind, ["s1", "cand", "ce"]]], ignore_index=True)
    d = d.drop_duplicates(["s1", "cand"])
    return pd.Series(d.ce.values.astype(np.float32), index=key(d.s1, d.cand)).reindex(key(s, c)).values


def combine(cols, fallback):
    M = np.vstack(cols)
    n = (~np.isnan(M)).sum(0)
    m = np.where(n > 0, np.nansum(M, 0) / np.maximum(n, 1), fallback)
    return m.astype(np.float32)


# cross-encoders available: ce1 = e5-base (all pairs), ce2 = e5-large, ce3 = LaBSE (uncertain pairs only)
tags = ["ce1"] + [t for t in ["ce2", "ce3", "ce4", "ce5", "ce6"] if ce_files(t, "val") and ce_files(t, "test")]
log(f"cross-encoders found: {tags}")
CV = {t: aligned(t, "val", Pv.s1, Pv.cand) for t in tags}
for t in tags:
    log(f"  {t}: val pairs covered {(~np.isnan(CV[t])).mean():.4f}")
ids = list(Pv.s1[Pv.is_seed].unique())
import itertools
best = None
for r in range(1, len(tags) + 1):
    for sub in itertools.combinations(tags, r):
        if "ce1" not in sub:  # ce2/ce3 cover only uncertain test pairs -> always keep ce1 as the base
            continue
        if a.tags and set(sub) != set(a.tags.split(",")):
            continue
        ce_v = combine([CV[t] for t in sub], pv)
        ws = [a.w] if a.w >= 0 else list(np.round(np.arange(0.0, 1.01, 0.1), 2))
        for w in ws:
            pb = w * pv + (1 - w) * ce_v
            (t_, f), _ = tune_threshold(Pv.s1.values, Pv.cand.values, pb, gt, ids)
            log(f"  CE={'+'.join(sub):<12} w_gbdt={w:.1f}: F0.5={f:.5f} thr={t_}")
            if best is None or f > best[0]:
                best = (f, w, t_, sub)
log(f"BEST blend: F0.5={best[0]:.5f} w_gbdt={best[1]} thr={best[2]} CE={'+'.join(best[3])}")

T = pd.read_pickle(f"{ART}/test_scored_stage1.pkl")
p = T.p.values.astype(np.float32)
ce_t = combine([aligned(t, "test", T.s1, T.cand) for t in best[3]], p)
log(f"test pairs {len(T)}, CE={'+'.join(best[3])}")
w, thr = best[1], best[2]
unseen = np.zeros(len(T), dtype=bool)
if b1.get("unseen") is not None and "country" in T.columns:
    unseen = ~np.isin(T.country.values, b1.get("train_countries", []))
pf = w * p + (1 - w) * ce_t
if unseen.any():
    tu = b1["unseen"]["thr"]
    base = pf if a.blend_unseen else p
    pf[unseen] = np.minimum(base[unseen] * (thr / tu), 0.999)
matches = postprocess(T.s1.values, T.cand.values, pf, thr)
s1_all = load_fields(a.work, "test", [1], None, None, cols=["entity_id"]).entity_id.values
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/matching_results.tsv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for s, ss in zip(s1_all, int_to_id(s1_all)):
        lst = matches.get(s, [])
        fh.write(ss + "\t" + (",".join(int_to_id(lst)) if len(lst) else "") + "\n")
log(f"wrote {a.out}/matching_results.tsv  ({sum(len(v) for v in matches.values())} matches)")
