"""Step 2: blocking / candidate generation.

Inverted index over multiple cheap keys (name token pairs, concatenated names, house-number +
street/city token, name-token + house-number), IDF-weighted key overlap score, top-M per S1.
Works per country (country treated as an open set of labels).

python src/block.py --work work --split train   (all train S1)
python src/block.py --work work --split test
"""
import argparse
import gc
import os
from multiprocessing import Pool

from common import *
from common import _part_no

ADDR_STOP = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "cir", "pl", "hwy", "pkwy", "no",
             "near", "po", "box", "fl", "h", "plot", "ste", "apt", "unit", "bldg", "opposite",
             "district", "taluk", "rue", "ch", "rte", "and", "of", "the", "de", "du", "des", "la",
             "le", "les", "pmb", "shop", "office", "sec", "block", "road", "street", "main",
             "cross", "n", "s", "e", "w", "village", "post", "tq", "dist", "city"}


_CAT = {"c": 0, "s": 0, "p": 0, "u": 0, "a": 1, "x": 2, "n": 2, "m": 2, "z": 2}


def p4(t):
    return t[:4]


def record_keys(core, nums, alph):
    core_t = core.split()
    nums_t = nums.split()[:4]
    alph_t = [t for t in alph.split() if len(t) >= 3 and t not in ADDR_STOP and not t.isdigit()][:8]
    keys = set()
    if core_t:
        keys.add("c:" + "".join(core_t))
        keys.add("s:" + " ".join(sorted(set(core_t))))
        ct = list(dict.fromkeys(core_t))[:6]
        pre = sorted(set(p4(t) for t in ct if len(t) >= 2))
        for i in range(len(pre)):
            keys.add("u:" + pre[i])
            for j in range(i + 1, len(pre)):
                keys.add("p:" + pre[i] + "|" + pre[j])
        for n in nums_t[:2]:
            for t in pre[:4]:
                keys.add("x:" + t + "|" + n)
    for n in nums_t[:3]:
        if len(n) > 7:
            continue
        for t in alph_t:
            keys.add("a:" + n + "|" + p4(t))
    if core_t and alph_t:
        cc = "".join(core_t)
        for t in alph_t:
            keys.add("n:" + cc + "|" + p4(t))
        cp = list(dict.fromkeys(p4(t) for t in core_t if len(t) >= 2))[:3]
        for t in cp:
            for u in alph_t[:6]:
                keys.add("m:" + t + "|" + p4(u))
    for n in nums_t:
        if len(n) in (5, 6):  # zip / pin
            for t in (core_t[:3]):
                keys.add("z:" + n + "|" + p4(t))
    return keys


def _keys_chunk(args):
    offset, cores, nums, alph = args
    ri, kk = [], []
    for i, (c, n, a) in enumerate(zip(cores, nums, alph)):
        ks = record_keys(c, n, a)
        ri.extend([offset + i] * len(ks))
        kk.extend(ks)
    h = pd.util.hash_array(np.array(kk, dtype=object), categorize=False)
    h32 = (h & np.uint64(0xFFFFFFFF))
    packed = (h32 << np.uint64(32)) | np.array(ri, dtype=np.uint64)
    return packed


def build_keys(df, workers):
    """returns sorted uint64 array: (hash32 << 32) | row_index"""
    cores, nums, alph = df.core.tolist(), df.nums.tolist(), df.alph.tolist()
    step = 20000
    tasks = [(i, cores[i:i + step], nums[i:i + step], alph[i:i + step]) for i in range(0, len(df), step)]
    out = []
    with Pool(workers) as pool:
        for pk in pool.imap(_keys_chunk, tasks, chunksize=1):
            out.append(pk)
    if not out:
        return np.zeros(0, np.uint64)
    arr = np.concatenate(out)
    del out
    arr.sort()
    return arr


def block_keys(ck, nC, qk, nQ, max_df, top_m, top_side=5):
    """ck/qk: sorted packed keys. returns DataFrame(q, c, bscore, nkeys, brank) (row indices)"""
    ch = (ck >> np.uint64(32)).astype(np.uint32)
    cri = (ck & np.uint64(0xFFFFFFFF)).astype(np.int32)
    del ck
    brk = np.flatnonzero(ch[1:] != ch[:-1]) + 1
    ustart = np.concatenate([[0], brk]).astype(np.int64)
    ucnt = np.diff(np.concatenate([ustart, [len(ch)]])).astype(np.int64)
    uk = ch[ustart]
    del ch, brk
    idf = np.log1p(nC / ucnt).astype(np.float32)
    qh = (qk >> np.uint64(32)).astype(np.uint32)
    qri = (qk & np.uint64(0xFFFFFFFF)).astype(np.int32)
    del qk
    o = np.argsort(qri, kind="stable")
    qri, qh = qri[o], qh[o]
    pos = np.searchsorted(uk, qh)
    pos[pos >= len(uk)] = 0
    hit = uk[pos] == qh
    cnt = np.where(hit, ucnt[pos], 0)
    ok = (cnt > 0) & (cnt <= max_df)
    qri, pos, cnt = qri[ok], pos[ok], cnt[ok]
    start = ustart[pos]
    w = idf[pos]
    kc = (uk[pos] >> np.uint32(29)).astype(np.int8)
    out = []
    # chunk boundaries on query rows so that each chunk expands to <= max_expand pairs
    max_expand = 12_000_000
    qstart = np.searchsorted(qri, np.arange(nQ + 1))
    ccum = np.concatenate([[0], np.cumsum(cnt)])
    per_q = ccum[qstart[1:]] - ccum[qstart[:-1]]
    qcum = np.cumsum(per_q)
    bnds_q = [0]
    while bnds_q[-1] < nQ:
        base = qcum[bnds_q[-1] - 1] if bnds_q[-1] > 0 else 0
        nxt = int(np.searchsorted(qcum, base + max_expand, side="right"))
        bnds_q.append(max(nxt, bnds_q[-1] + 1))
    bnds_q[-1] = min(bnds_q[-1], nQ)
    bounds = qstart[np.array(bnds_q)]
    for b in range(len(bounds) - 1):
        a0, a1 = bounds[b], bounds[b + 1]
        if a1 <= a0:
            continue
        cc = cnt[a0:a1]
        tot = int(cc.sum())
        rep_q = np.repeat(qri[a0:a1], cc)
        rep_w = np.repeat(w[a0:a1], cc)
        rep_k = np.repeat(kc[a0:a1], cc)
        csum = np.cumsum(cc) - cc
        idx = np.arange(tot, dtype=np.int64) - np.repeat(csum, cc) + np.repeat(start[a0:a1], cc)
        rep_c = cri[idx]
        del idx
        code = rep_q.astype(np.int64) * nC + rep_c
        del rep_q, rep_c
        o = np.argsort(code, kind="stable")
        code, rep_w, rep_k = code[o], rep_w[o], rep_k[o]
        brk = np.flatnonzero(code[1:] != code[:-1]) + 1
        st = np.concatenate([[0], brk])
        uc = code[st]
        sc = np.add.reduceat(rep_w, st)
        sn = np.add.reduceat(np.where(rep_k == 0, rep_w, 0), st)
        sa = np.add.reduceat(np.where(rep_k == 1, rep_w, 0), st)
        nk = np.diff(np.append(st, len(code))).astype(np.int16)
        qq = (uc // nC).astype(np.int32)
        cc2 = (uc % nC).astype(np.int32)

        def ranks(score):
            o = np.lexsort((-score, qq))
            r = np.empty(len(qq), np.int32)
            q_s = qq[o]
            first = np.searchsorted(q_s, q_s, side="left")
            r[o] = np.arange(len(qq)) - first
            return r
        r_t, r_n, r_a = ranks(sc), ranks(sn), ranks(sa)
        keep = (r_t < top_m) | ((r_n < top_side) & (sn > 0)) | ((r_a < top_side) & (sa > 0))
        out.append(pd.DataFrame({"q": qq[keep], "c": cc2[keep], "bscore": sc[keep].astype(np.float32),
                                 "bname": sn[keep].astype(np.float32), "baddr": sa[keep].astype(np.float32),
                                 "nkeys": nk[keep], "brank": r_t[keep].astype(np.int16),
                                 "rank_n": r_n[keep].astype(np.int16), "rank_a": r_a[keep].astype(np.int16)}))
    if not out:
        return pd.DataFrame({c: [] for c in ["q", "c", "bscore", "bname", "baddr", "nkeys", "brank", "rank_n", "rank_a"]})
    return pd.concat(out, ignore_index=True)


def id_to_int(ids):
    return np.array([int(x[1]) * 10 ** 10 + int(x[3:]) for x in ids], dtype=np.int64)


def int_to_id(v):
    return np.array([f"S{x // 10 ** 10}-{x % 10 ** 10}" for x in v.tolist()], dtype=object)


def part_files(work, split, k):
    import glob
    fs = glob.glob(f"{work}/norm_{split}_s{k}.p*.pkl") + glob.glob(f"{work}/norm_{split}_s{k}.p*.parquet")
    return sorted(fs, key=_part_no)


def _read_part(path, cols):
    from data import read_part
    return read_part(path, cols)


def _part_keys(args):
    path, country, subset = args
    df = _read_part(path, ["entity_id", "country", "core", "nums", "alph"])
    df = df[df.country == country]
    if subset is not None:
        df = df[df.entity_id.isin(subset)]
    ids = id_to_int(df.entity_id.values)
    core, nums, alph = df.core.values, df.nums.values, df.alph.values
    del df
    out = []
    step = 50000
    for s0 in range(0, len(ids), step):
        ri, kk = [], []
        for i in range(s0, min(s0 + step, len(ids))):
            ks = record_keys(core[i], nums[i], alph[i])
            ri.extend([i] * len(ks))
            kk.extend(ks)
        if kk:
            h = pd.util.hash_array(np.array(kk, dtype=object), categorize=False)
            cat = np.array([_CAT.get(k[0], 2) for k in kk], dtype=np.uint64)
            h = (h & np.uint64(0x1FFFFFFF)) | (cat << np.uint64(29))
            out.append((h << np.uint64(32)) | np.array(ri, dtype=np.uint64))
    packed = np.concatenate(out) if out else np.zeros(0, np.uint64)
    return ids, packed


def keys_from_parts(files, country, workers, subset=None):
    ids_all, keys_all, off = [], [], 0
    with Pool(workers) as pool:
        for ids, packed in pool.imap(_part_keys, [(f, country, subset) for f in files], chunksize=1):
            if len(ids):
                packed = packed + np.uint64(off)  # row index lives in the low 32 bits
                ids_all.append(ids); keys_all.append(packed)
                off += len(ids)
    if not ids_all:
        return np.zeros(0, np.int64), np.zeros(0, np.uint64)
    keys = np.concatenate(keys_all); del keys_all
    keys.sort()
    return np.concatenate(ids_all), keys


def countries_of(work, split):
    cs = set()
    for f in part_files(work, split, 1):
        cs |= set(_read_part(f, ["country"]).country.unique())
    return sorted(cs)


def run(work, split, workers, max_df, top_m, s1_subset=None, top_side=5):
    res = []
    for country in countries_of(work, split):
        cfiles = part_files(work, split, 2) + part_files(work, split, 3)
        c_ids, ck = keys_from_parts(cfiles, country, workers)
        q_ids, qk = keys_from_parts(part_files(work, split, 1), country, workers, s1_subset)
        log(f"[{split}] country={country}  S1={len(q_ids)}  S2+S3={len(c_ids)}  keys={len(ck)}/{len(qk)}")
        if len(c_ids) == 0 or len(q_ids) == 0:
            continue
        pr = block_keys(ck, len(c_ids), qk, len(q_ids), max_df, top_m, top_side)
        del ck, qk
        pr["s1"] = q_ids[pr.q.values]
        pr["cand"] = c_ids[pr.c.values]
        res.append(pr.drop(columns=["q", "c"]))
        log(f"   pairs={len(pr)}")
        del pr
        gc.collect()
    return pd.concat(res, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count()))
    ap.add_argument("--max_df", type=int, default=300)
    ap.add_argument("--top_m", type=int, default=12)
    ap.add_argument("--top_side", type=int, default=3)
    ap.add_argument("--subset_frac", type=float, default=1.0)
    a = ap.parse_args()
    sub = None
    if a.subset_frac < 1.0:
        s1ids = load_df(f"{a.work}/norm_{a.split}_s1", ["entity_id"]).entity_id
        sub = set(s1ids.sample(frac=a.subset_frac, random_state=0))
    out = run(a.work, a.split, a.workers, a.max_df, a.top_m, sub, a.top_side)
    tag = "" if a.subset_frac >= 1.0 else f"_sub{a.subset_frac}"
    save_df(out, f"{a.work}/blocks_{a.split}{tag}")
    if sub is not None:
        np.save(f"{a.work}/blocks_{a.split}{tag}_s1.npy", id_to_int(sorted(sub)))
    log("saved", len(out), "pairs")
