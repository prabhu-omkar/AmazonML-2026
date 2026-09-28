"""Step 1: learn transliteration map from train GT and normalise every source file.

python src/prepare.py --data dataset --work work
"""
import argparse, json, os, random
from multiprocessing import Pool
import csv
from common import *

_NORM = None


def _init(tok_map):
    global _NORM
    _NORM = Normalizer(tok_map)


def _norm_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nm, toks, core, is_dom, has_dba = _NORM.name(n)
        ad, atoks, nums, alph = _NORM.address(a)
        out.append((nm, " ".join(core), ad, " ".join(nums), " ".join(alph), is_dom, has_dba))
    return out


def normalise_file(path, tok_map, workers, out_path, part_rows=1000000):
    parts = 0
    reader = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                         na_filter=False, encoding="utf-8", chunksize=part_rows)
    with Pool(workers, initializer=_init, initargs=(tok_map,)) as pool:
        for df in reader:
            df = df.astype(object)
            names, addrs = df.business_name.tolist(), df.business_address.tolist()
            step = 20000
            chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
            res = []
            for r in pool.imap(_norm_chunk, chunks, chunksize=1):
                res.extend(r)
            out = pd.DataFrame(res, columns=["nm", "core", "ad", "nums", "alph", "is_dom", "has_dba"])
            del res
            out.insert(0, "country", df.country.values)
            out.insert(0, "entity_id", df.entity_id.values)
            out["is_dom"] = out.is_dom.astype(np.int8)
            out["has_dba"] = out.has_dba.astype(np.int8)
            save_df(out, f"{out_path}.p{parts}")
            parts += 1
            del out, df
    return parts


def learn_map(data, n_entities=400000, seed=0):
    s1 = read_tsv(f"{data}/train/train_source1.tsv")
    gt = read_ground_truth(f"{data}/train/train_ground_truth.tsv")
    random.seed(seed)
    keys = random.sample(list(gt), min(n_entities, len(gt)))
    want = set(x for k in keys for x in gt[k])
    s1i = dict(zip(s1.entity_id, zip(s1.business_name, s1.business_address)))
    rec = {}
    for f in ["train_source2", "train_source3"]:
        d = read_tsv(f"{data}/train/{f}.tsv")
        d = d[d.entity_id.isin(want)]
        rec.update(zip(d.entity_id, zip(d.business_name, d.business_address)))
    npairs, apairs = [], []
    for k in keys:
        for x in gt[k]:
            if x in rec:
                npairs.append((s1i[k][0], rec[x][0]))
                apairs.append((s1i[k][1], rec[x][1]))
    nm = learn_token_map(npairs)
    am = learn_token_map(apairs)
    tm = dict(am)
    tm.update(nm)
    return tm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    mp = f"{a.work}/token_map.json"
    if not os.path.exists(mp):
        log("learning transliteration token map")
        tm = learn_map(a.data)
        json.dump(tm, open(mp, "w", encoding="utf-8"), ensure_ascii=False)
    tm = json.load(open(mp, encoding="utf-8"))
    log("token map size", len(tm))
    for split in a.splits.split(","):
        for k in (1, 2, 3):
            out = f"{a.work}/norm_{split}_s{k}"
            if exists_df(out):
                continue
            log("normalising", split, k)
            n = normalise_file(f"{a.data}/{split}/{split}_source{k}.tsv", tm, a.workers, out + ".tmp")
            import glob
            for f in glob.glob(out + ".tmp.p*"):
                os.rename(f, f.replace(".tmp.p", ".p"))
            log("  parts", n)
