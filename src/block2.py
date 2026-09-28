"""v7 candidate generation: large pool -> learned ranker -> variable-size candidate sets.

Stages (run in this order):
  python src/block2.py pool  --work W --split train      # per country: key blocking (top-30 + side slots)
  python src/block2.py pool  --work W --split test       #   + query expansion from the top-3 candidates
                                                         #   + group/competition features -> pool_{split}_{country}.parquet
  python src/block2.py fit   --work W --data DATA        # ranker (LightGBM) on 150k non-validation train S1,
                                                         #   recall/ceiling report on the 40k validation S1
  python src/block2.py apply --work W --split train      # score every pool pair, keep p >= cutoff where the
  python src/block2.py apply --work W --split test       #   cutoff gives --budget candidates per S1 on average
                                                         #   -> blocks_{split}_v7.parquet (rp = ranker prob)
Then:  train.py --blocks blocks_train_v7 --K 1000 --side 0 ;  predict.py --blocks blocks_test_v7
"""
import argparse
import gc
import pickle
from multiprocessing import Pool

from common import *
from block import record_keys, part_files, countries_of, _read_part, id_to_int

M32 = np.uint64(0xFFFFFFFF)
KTYPE = {"c": 0, "s": 1, "p": 2, "u": 3, "n": 4, "m": 5, "a": 6, "x": 7, "z": 8}
# channel of each key type (0 = name, 1 = address, 2 = mixed) - as in block.py
KCAT = np.array([0, 0, 0, 0, 2, 2, 1, 2, 2], dtype=np.int8)
# feature groups: name-exact(c,s) name-fuzzy(p,u) name x locality(n,m) address(a) name x number(x) zip(z)
KGRP = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5], dtype=np.int8)
NGRP = 6


# ----------------------------------------------------------------------------- keys / index
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
    for s0 in range(0, len(ids), 50000):
        ri, kk = [], []
        for i in range(s0, min(s0 + 50000, len(ids))):
            ks = record_keys(core[i], nums[i], alph[i])
            ri.extend([i] * len(ks))
            kk.extend(ks)
        if kk:
            h = pd.util.hash_array(np.array(kk, dtype=object), categorize=False)
            t = np.array([KTYPE[k[0]] for k in kk], dtype=np.uint64)
            h = (h & np.uint64(0x0FFFFFFF)) | (t << np.uint64(28))
            out.append((h << np.uint64(32)) | np.array(ri, dtype=np.uint64))
    return ids, (np.concatenate(out) if out else np.zeros(0, np.uint64))


def keys_from_parts(files, country, workers, subset=None):
    ids_all, keys_all, off = [], [], 0
    with Pool(workers) as pool:
        for ids, packed in pool.imap(_part_keys, [(f, country, subset) for f in files], chunksize=1):
            if len(ids):
                ids_all.append(ids)
                keys_all.append(packed + np.uint64(off))
                off += len(ids)
    if not ids_all:
        return np.zeros(0, np.int64), np.zeros(0, np.uint64)
    keys = np.concatenate(keys_all)
    del keys_all
    keys.sort()
    return np.concatenate(ids_all), keys


def build_index(ck, nC):
    ch = (ck >> np.uint64(32)).astype(np.uint32)
    brk = np.flatnonzero(ch[1:] != ch[:-1]) + 1
    ustart = np.concatenate([[0], brk]).astype(np.int64)
    del brk
    ucnt = np.diff(np.concatenate([ustart, [len(ch)]])).astype(np.int32)
    uk = ch[ustart]
    del ch
    cri = (ck & M32).astype(np.int32)
    return dict(uk=uk, ustart=ustart, ucnt=ucnt, idf=np.log1p(nC / ucnt).astype(np.float32), cri=cri, nC=nC)


def select_rows(packed, look, step=10_000_000):
    """look: int64 array row -> new id (-1 = skip). returns (hash32, new_id)"""
    H, R = [], []
    for i in range(0, len(packed), step):
        p = packed[i:i + step]
        r = look[(p & M32).astype(np.int64)]
        sel = r >= 0
        H.append((p[sel] >> np.uint64(32)).astype(np.uint32))
        R.append(r[sel].astype(np.int32))
    return np.concatenate(H), np.concatenate(R)


def _ranks(qq, score):
    o = np.lexsort((-score, qq))
    r = np.empty(len(qq), np.int32)
    q_s = qq[o]
    r[o] = np.arange(len(qq)) - np.searchsorted(q_s, q_s, side="left")
    return r


def query(ix, qh, qri, nQ, max_df, top_m, top_side, detail=True, max_expand=12_000_000, qcap=None):
    """score all (query row, candidate row) pairs sharing a key with df <= max_df; keep top_m by total
    score plus top_side by name-only / address-only score."""
    uk, ustart, ucnt, idf, cri, nC = ix["uk"], ix["ustart"], ix["ucnt"], ix["idf"], ix["cri"], ix["nC"]
    o = np.argsort(qri, kind="stable")
    qri, qh = qri[o], qh[o]
    pos = np.searchsorted(uk, qh)
    pos[pos >= len(uk)] = 0
    cnt = np.where(uk[pos] == qh, ucnt[pos], 0)
    ok = (cnt > 0) & (cnt <= (max_df if qcap is None else qcap[qri]))
    qri, pos, cnt = qri[ok], pos[ok], cnt[ok].astype(np.int64)
    start, w = ustart[pos], idf[pos]
    kt = (uk[pos] >> np.uint32(28)).astype(np.int8)
    del pos, ok
    qstart = np.searchsorted(qri, np.arange(nQ + 1))
    ccum = np.concatenate([[0], np.cumsum(cnt)])
    cq = ccum[qstart]
    out = []
    b0 = 0
    while b0 < nQ:
        b1 = int(np.searchsorted(cq, cq[b0] + max_expand, side="right")) - 1
        b1 = min(max(b1, b0 + 1), nQ)
        a0, a1 = qstart[b0], qstart[b1]
        b0 = b1
        if a1 <= a0:
            continue
        cc = cnt[a0:a1]
        tot = int(cc.sum())
        rep_w = np.repeat(w[a0:a1], cc)
        rep_t = np.repeat(kt[a0:a1], cc)
        csum = np.cumsum(cc) - cc
        idx = np.arange(tot, dtype=np.int64) - np.repeat(csum, cc) + np.repeat(start[a0:a1], cc)
        code = np.repeat(qri[a0:a1], cc).astype(np.int64) * nC + cri[idx]
        del idx
        o = np.argsort(code, kind="stable")
        code, rep_w, rep_t = code[o], rep_w[o], rep_t[o]
        st = np.concatenate([[0], np.flatnonzero(code[1:] != code[:-1]) + 1])
        qq = (code[st] // nC).astype(np.int32)
        c2 = (code[st] % nC).astype(np.int32)
        del code
        cat = KCAT[rep_t]
        sc = np.add.reduceat(rep_w, st)
        sn = np.add.reduceat(np.where(cat == 0, rep_w, 0), st)
        sa = np.add.reduceat(np.where(cat == 1, rep_w, 0), st)
        rt = _ranks(qq, sc)
        keep = rt < top_m
        if top_side:
            keep |= ((_ranks(qq, sn) < top_side) & (sn > 0)) | ((_ranks(qq, sa) < top_side) & (sa > 0))
        d = {"q": qq[keep], "c": c2[keep], "sc": sc[keep].astype(np.float32)}
        if detail:
            d["sn"] = sn[keep].astype(np.float32)
            d["sa"] = sa[keep].astype(np.float32)
            d["nk"] = np.diff(np.append(st, len(rep_w)))[keep].astype(np.int16)
            d["maxidf"] = np.maximum.reduceat(rep_w, st)[keep].astype(np.float32)
            grp = KGRP[rep_t]
            for g in range(NGRP):
                d[f"g{g}"] = np.add.reduceat((grp == g).astype(np.int16), st)[keep].astype(np.int8)
        out.append(pd.DataFrame(d))
        del rep_w, rep_t, cat, st
    if not out:
        return pd.DataFrame(columns=["q", "c", "sc"])
    return pd.concat(out, ignore_index=True)


def group_rank(keys, score):
    return _ranks(keys.astype(np.int64), score.astype(np.float64)).astype(np.int16)


# ----------------------------------------------------------------------------- stage: pool
def build_pool(work, split, country, workers, top_m, side, n_exp, n_seed, max_df, s1_subset=None, max_df_noaddr=0):
    cfiles = part_files(work, split, 2) + part_files(work, split, 3)
    c_ids, ck = keys_from_parts(cfiles, country, workers)
    q_ids, qk = keys_from_parts(part_files(work, split, 1), country, workers, s1_subset)
    nC, nQ = len(c_ids), len(q_ids)
    log(f"[{split}/{country}] S1={nQ} S2+S3={nC} keys={len(ck)}/{len(qk)}")
    ix = build_index(ck, nC)
    qcap = None
    if max_df_noaddr > max_df:
        # S1 records without address keys: their name keys are all they have - allow more frequent keys
        qt = ((qk >> np.uint64(60)) & np.uint64(0xF)).astype(np.int8)
        has_addr = np.zeros(nQ, bool)
        has_addr[(qk[(qt == 4) | (qt == 5) | (qt == 6)] & M32).astype(np.int64)] = True
        qcap = np.where(has_addr, max_df, max_df_noaddr).astype(np.int64)
        log(f"   S1 without address keys: {(~has_addr).sum()} (df cap {max_df_noaddr})")
    P = query(ix, (qk >> np.uint64(32)).astype(np.uint32), (qk & M32).astype(np.int32), nQ, max_df, top_m, side,
              qcap=qcap)
    del qk
    gc.collect()
    log(f"   direct pool {len(P)} ({len(P) / max(nQ, 1):.1f}/S1)")
    # ---- query expansion: re-query with the keys of each S1's top-n_seed candidates
    P["r"] = group_rank(P.q.values, P.sc.values)
    S = P.loc[P.r < n_seed, ["q", "c"]]
    urows = np.unique(S.c.values)
    look = np.full(nC, -1, np.int64)
    look[urows] = np.arange(len(urows))
    eh, er = select_rows(ck, look)
    E = query(ix, eh, er, len(urows), max_df, 10, 0, detail=False)
    del eh, er, ix, ck
    gc.collect()
    E["c_seed"] = urows[E.q.values]
    E = E[E.c_seed.values != E.c.values]
    J = S.rename(columns={"c": "c_seed"}).merge(E[["c_seed", "c", "sc"]], on="c_seed")
    del E, S
    code = J.q.values.astype(np.int64) * nC + J.c.values
    J = pd.DataFrame({"code": code, "esc": J.sc.values}).groupby("code").esc.agg(["max", "size"])
    pc = P.q.values.astype(np.int64) * nC + P.c.values
    inpool = np.isin(J.index.values, pc)
    Jp = J[inpool]
    m = pd.Series(Jp["max"].values, index=Jp.index)
    P["esc"] = m.reindex(pc).fillna(0).values.astype(np.float32)
    P["en"] = pd.Series(Jp["size"].values, index=Jp.index).reindex(pc).fillna(0).values.astype(np.int8)
    Jn = J[~inpool]
    N = pd.DataFrame({"q": (Jn.index.values // nC).astype(np.int32), "c": (Jn.index.values % nC).astype(np.int32),
                      "esc": Jn["max"].values.astype(np.float32), "en": Jn["size"].values.astype(np.int8)})
    N = N[group_rank(N.q.values, N.esc.values) < n_exp]
    log(f"   expansion adds {len(N)} ({len(N) / max(nQ, 1):.2f}/S1)")
    for col in ["sc", "sn", "sa", "maxidf"]:
        N[col] = np.float32(0)
    for col in ["nk"] + [f"g{g}" for g in range(NGRP)]:
        N[col] = 0
    P = pd.concat([P.drop(columns=["r"]), N], ignore_index=True)
    P["from_exp"] = np.r_[np.zeros(len(P) - len(N), np.int8), np.ones(len(N), np.int8)]
    del J, Jp, Jn, N, pc
    gc.collect()
    # ---- group / competition features (need every S1 of the split+country)
    q, c = P.q.values, P.c.values
    mx = pd.Series(P.sc.values).groupby(q).transform("max").values
    P["b_rel"] = (P.sc.values / np.maximum(mx, 1e-6)).astype(np.float32)
    P["b_gap"] = (mx - P.sc.values).astype(np.float32)
    P["brank"] = group_rank(q, P.sc.values)
    P["rank_n"] = group_rank(q, P.sn.values)
    P["rank_a"] = group_rank(q, P.sa.values)
    P["erank"] = group_rank(q, P.esc.values)
    P["n_pool"] = pd.Series(q).groupby(q).transform("size").values.astype(np.int16)
    gc_ = pd.Series(P.sc.values).groupby(c)
    P["c_nq"] = gc_.transform("size").values.astype(np.int16)
    P["c_rank"] = group_rank(c, P.sc.values)
    cmax = gc_.transform("max").values
    # best OTHER claimant's score: second max if this row is the max
    o = np.lexsort((-P.sc.values, c))
    c_s = c[o]
    first = np.searchsorted(c_s, c_s, side="left")
    sc_s = P.sc.values[o]
    second = np.where(first + 1 < len(c_s), sc_s[np.minimum(first + 1, len(c_s) - 1)], 0)
    second = np.where((first + 1 < len(c_s)) & (c_s[np.minimum(first + 1, len(c_s) - 1)] == c_s), second, 0)
    sec = np.empty(len(P), np.float32)
    sec[o] = second
    P["c_gap"] = (P.sc.values - np.where(P.sc.values >= cmax, sec, cmax)).astype(np.float32)
    P["s1"] = q_ids[q]
    P["cand"] = c_ids[c]
    return P.drop(columns=["c"])


# ----------------------------------------------------------------------------- text features
def _fnum(nums):
    t = nums.split()
    return t[0] if t else ""


def _zip(nums):
    for t in nums.split():
        if len(t) in (5, 6) and t.isdigit():
            return t
    return ""


def record_table(work, split, ks, country):
    d = load_fields_(work, split, ks, country)
    core = d.core.fillna("").values
    nums = d.nums.fillna("").values
    T = pd.DataFrame({
        "core": core,
        "core_h": pd.util.hash_array(core.astype(object), categorize=False).astype(np.int64),
        "fnum": pd.util.hash_array(np.array([_fnum(x) for x in nums], dtype=object), categorize=False).astype(np.int64),
        "has_num": np.array([len(x) > 0 for x in nums], dtype=np.int8),
        "zip": pd.util.hash_array(np.array([_zip(x) for x in nums], dtype=object), categorize=False).astype(np.int64),
        "has_zip": np.array([len(_zip(x)) > 0 for x in nums], dtype=np.int8),
        "ad_empty": (d.ad.fillna("").str.len().values == 0).astype(np.int8),
        "ntok": np.array([len(x.split()) for x in core], dtype=np.int8),
    }, index=d.entity_id.values)
    T["core_freq"] = T.core_h.map(T.core_h.value_counts()).values.astype(np.int32)
    return T


def load_fields_(work, split, ks, country):
    out = []
    for k in ks:
        for f in part_files(work, split, k):
            d = _read_part(f, ["entity_id", "country", "core", "nums", "ad"])
            d = d[d.country == country].copy()
            d["entity_id"] = id_to_int(d.entity_id.values)
            out.append(d.drop(columns=["country"]))
    return pd.concat(out, ignore_index=True)


def _jw_py(a, b):
    """pure-python Jaro-Winkler fallback (only used where rapidfuzz is unavailable)"""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    r = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - r), min(lb, i + r + 1)):
            if not mb[j] and b[j] == ch:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    j = (m / la + m / lb + (m - t / 2) / m) / 3
    p = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        p += 1
    return j + p * 0.1 * (1 - j)


def text_sims(A, B):
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import JaroWinkler
        from rapidfuzz import fuzz
        jw = process.cpdist(A, B, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
        ts = (process.cpdist(A, B, scorer=fuzz.token_set_ratio, workers=-1) / 100.0).astype(np.float32)
        return jw, ts
    except ImportError:
        jw = np.array([_jw_py(a, b) for a, b in zip(A, B)], dtype=np.float32)
        ts = np.array([len(set(a.split()) & set(b.split())) / max(1, min(len(a.split()), len(b.split())))
                       for a, b in zip(A, B)], dtype=np.float32)
        return jw, ts


def text_features(P, T1, T2):
    i1 = T1.index.get_indexer(P.s1.values)
    i2 = T2.index.get_indexer(P.cand.values)
    F = {}
    a = T1.iloc[i1]
    b = T2.iloc[i2]
    F["core_eq"] = (a.core_h.values == b.core_h.values).astype(np.int8)
    F["fnum_eq"] = np.where((a.has_num.values == 1) & (b.has_num.values == 1),
                            (a.fnum.values == b.fnum.values).astype(np.int8), -1).astype(np.int8)
    F["zip_eq"] = np.where((a.has_zip.values == 1) & (b.has_zip.values == 1),
                           (a.zip.values == b.zip.values).astype(np.int8), -1).astype(np.int8)
    F["ad_empty1"] = a.ad_empty.values
    F["ad_empty2"] = b.ad_empty.values
    F["ntok1"] = a.ntok.values
    F["ntok2"] = b.ntok.values
    F["core_freq1"] = a.core_freq.values
    F["core_freq2"] = b.core_freq.values
    F["src3"] = (P.cand.values // 10 ** 10 == 3).astype(np.int8)
    jw, ts = text_sims(list(a.core.values), list(b.core.values))
    F["jw_core"], F["tset_core"] = jw, ts
    F = pd.DataFrame(F, index=P.index)
    F["jw_rank"] = group_rank(P.s1.values, F.jw_core.values + 0.001 * P.b_rel.values)
    F["jw_rel"] = (F.jw_core.values - pd.Series(F.jw_core.values).groupby(P.s1.values).transform("max").values).astype(np.float32)
    return F


RANK_COLS = ["sc", "sn", "sa", "nk", "maxidf"] + [f"g{g}" for g in range(NGRP)] + \
    ["esc", "en", "from_exp", "b_rel", "b_gap", "brank", "rank_n", "rank_a", "erank", "n_pool",
     "c_nq", "c_rank", "c_gap", "core_eq", "fnum_eq", "zip_eq", "ad_empty1", "ad_empty2", "ntok1", "ntok2",
     "core_freq1", "core_freq2", "src3", "jw_core", "tset_core", "jw_rank", "jw_rel"]


try:
    import pyarrow  # noqa: F401
    HAS_PA = True
except ImportError:  # container fallback: pickle
    HAS_PA = False


def pool_path(work, split, country):
    return f"{work}/pool_{split}_{country}." + ("parquet" if HAS_PA else "pkl")


def save_pool(P, work, split, country):
    if HAS_PA:
        P.to_parquet(pool_path(work, split, country), index=False, row_group_size=1_000_000)
    else:
        P.to_pickle(pool_path(work, split, country))


def iter_pool(work, split, country, s1_keep=None, q_step=150000):
    if not HAS_PA:
        P = pd.read_pickle(pool_path(work, split, country))
        if s1_keep is not None:
            P = P[np.isin(P.s1.values, s1_keep)]
        for lo in range(0, int(P.q.max()) + 1, q_step):
            Pc = P[(P.q.values >= lo) & (P.q.values < lo + q_step)]
            if len(Pc):
                yield Pc.reset_index(drop=True)
        return
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    dset = ds.dataset(pool_path(work, split, country), format="parquet")
    nq = int(pc.max(dset.to_table(columns=["q"]).column("q")).as_py()) + 1
    for lo in range(0, nq, q_step):
        flt = (ds.field("q") >= lo) & (ds.field("q") < lo + q_step)
        if s1_keep is not None:
            flt = flt & ds.field("s1").isin(pa_array(s1_keep))
        Pc = dset.to_table(filter=flt).to_pandas()
        if len(Pc):
            yield Pc


def iter_features(work, split, country, s1_keep=None, q_step=150000):
    """yields feature frames for consecutive S1 blocks (all rows of an S1 are in the same frame)"""
    T1 = record_table(work, split, [1], country)
    T2 = record_table(work, split, [2, 3], country)
    for Pc in iter_pool(work, split, country, s1_keep, q_step):
        yield pd.concat([Pc, text_features(Pc, T1, T2)], axis=1)


def pa_array(x):
    import pyarrow as pa
    return pa.array(np.asarray(x, dtype=np.int64))


def featurize_country(work, split, country, s1_keep=None):
    return pd.concat(list(iter_features(work, split, country, s1_keep)), ignore_index=True)


def val_ids(work, n_val=40000):
    from data import load_fields
    s1_meta = load_fields(work, "train", [1], None, None, cols=["entity_id", "country"])
    perm = s1_meta.entity_id.values[np.random.RandomState(42).permutation(len(s1_meta))]
    return perm[:n_val], perm[n_val:]


def ceiling(gt, sets, ids):
    tot = 0.0
    for s in ids:
        g = gt.get(s, set())
        if not g:
            tot += 1.0
            continue
        r = len(g & sets.get(s, set())) / len(g)
        tot += 1.25 * r / (0.25 + r) if r > 0 else 0.0
    return tot / len(ids)


def ranker_score(mdl, X):
    """raw (log-odds) score: probabilities of a confident GBDT underflow to exactly 0/1, which makes ties"""
    if hasattr(mdl, "booster_"):
        return mdl.predict(X, raw_score=True).astype(np.float32)
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(X).astype(np.float32)
    p = np.clip(mdl.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).astype(np.float32)


def select(P, p, budget, n_s1, min_keep=1):
    """keep rows with p >= cutoff, cutoff chosen so that avg kept per S1 == budget; always keep top min_keep."""
    r = group_rank(P.s1.values, p)
    forced = r < min_keep
    rest = np.sort(p[~forced])[::-1]
    n_target = int(budget * n_s1) - int(forced.sum())
    thr = rest[min(max(n_target, 1), len(rest)) - 1] if len(rest) else 1.0
    return forced | (p >= thr), float(thr)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "fit", "apply"])
    ap.add_argument("--work", default="work3")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count()))
    ap.add_argument("--top_m", type=int, default=30)
    ap.add_argument("--side", type=int, default=3)
    ap.add_argument("--n_exp", type=int, default=5)
    ap.add_argument("--n_seed", type=int, default=3)
    ap.add_argument("--max_df", type=int, default=300)
    ap.add_argument("--max_df_noaddr", type=int, default=0)
    ap.add_argument("--budget", type=float, default=13.0)
    ap.add_argument("--n_fit", type=int, default=150000)
    ap.add_argument("--s1_frac", type=float, default=1.0, help="testing only: restrict S1 set")
    ap.add_argument("--country", default="")
    ap.add_argument("--eval_present", type=int, default=0, help="testing only: evaluate only S1 present in pools")
    a = ap.parse_args()
    os.makedirs(f"{a.work}/models", exist_ok=True)

    if a.stage == "pool":
        sub = None
        if a.s1_frac < 1.0:
            ids = load_df(f"{a.work}/norm_{a.split}_s1", ["entity_id"]).entity_id
            sub = set(ids.sample(frac=a.s1_frac, random_state=0))
        for country in ([a.country] if a.country else countries_of(a.work, a.split)):
            P = build_pool(a.work, a.split, country, a.workers, a.top_m, a.side, a.n_exp, a.n_seed, a.max_df, sub,
                           a.max_df_noaddr)
            save_pool(P, a.work, a.split, country)
            log(f"   saved pool {len(P)} pairs")
            del P
            gc.collect()

    elif a.stage == "fit":
        from data import gt_int
        gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
        vids, rest = val_ids(a.work)
        rng = np.random.RandomState(7)
        fit_ids = rng.choice(rest, min(a.n_fit, len(rest)), replace=False)
        F_fit, F_val = [], []
        for country in countries_of(a.work, "train"):
            if not os.path.exists(pool_path(a.work, "train", country)):
                continue
            F = featurize_country(a.work, "train", country, np.concatenate([fit_ids, vids]))
            F["y"] = np.array([c in gt.get(s, ()) for s, c in zip(F.s1.values, F.cand.values)], dtype=np.int8)
            F_fit.append(F[np.isin(F.s1.values, fit_ids)])
            F_val.append(F[np.isin(F.s1.values, vids)])
            log(f"   featurised {country}: {len(F)} rows")
        F_fit = pd.concat(F_fit, ignore_index=True)
        F_val = pd.concat(F_val, ignore_index=True)
        if a.eval_present:
            vids = np.unique(F_val.s1.values)
        log(f"ranker train rows {len(F_fit)} pos {F_fit.y.mean():.3f}; val rows {len(F_val)}")
        try:
            import lightgbm as lgb
            mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=127, min_child_samples=50,
                                     min_child_weight=1.0, max_delta_step=2.0,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=a.workers,
                                     verbose=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=127)
        mdl.fit(F_fit[RANK_COLS].values.astype(np.float32), F_fit.y.values)
        pickle.dump({"model": mdl, "cols": RANK_COLS}, open(f"{a.work}/models/ranker.pkl", "wb"))
        # ---- report on validation S1 (never used for fitting)
        p = ranker_score(mdl, F_val[RANK_COLS].values.astype(np.float32))
        log(f"ranker score range {p.min():.2f}..{p.max():.2f}, distinct values {len(np.unique(p))}")
        vset = [s for s in vids if s in set(F_val.s1.values)] if False else list(vids)
        tot = sum(len(gt.get(s, ())) for s in vids)
        allsets = F_val.groupby("s1").cand.apply(set).to_dict()
        rec_all = sum(len(gt.get(s, set()) & allsets.get(s, set())) for s in vids) / tot
        log(f"VAL pool: {len(F_val) / len(vids):.1f} cand/S1 recall {rec_all:.4f}")
        base = F_val[(F_val.brank < 12) | ((F_val.rank_n < 3) & (F_val.sn > 0)) | ((F_val.rank_a < 3) & (F_val.sa > 0))]
        base = base[base.from_exp == 0]
        bs = base.groupby("s1").cand.apply(set).to_dict()
        rb = sum(len(gt.get(s, set()) & bs.get(s, set())) for s in vids) / tot
        log(f"VAL old rule (K=12+3+3): {len(base) / len(vids):.2f} cand/S1 recall {rb:.4f} ceiling {ceiling(gt, bs, vids):.4f}")
        for B in [8, 10, 13, 16, 20]:
            keep, thr = select(F_val, p, B, len(vids))
            ks = F_val[keep].groupby("s1").cand.apply(set).to_dict()
            r = sum(len(gt.get(s, set()) & ks.get(s, set())) for s in vids) / tot
            log(f"VAL ranker budget {B:>2}: {keep.sum() / len(vids):.2f} cand/S1 recall {r:.4f} "
                f"ceiling {ceiling(gt, ks, vids):.4f} (cutoff {thr:.3f})")
        imp = getattr(mdl, "feature_importances_", None)
        if imp is not None:
            for c_, v in sorted(zip(RANK_COLS, imp), key=lambda x: -x[1])[:15]:
                log(f"   imp {c_}: {v}")

    elif a.stage == "apply":
        R = pickle.load(open(f"{a.work}/models/ranker.pkl", "rb"))
        parts, probs, n_s1 = [], [], 0
        keepcols = ["s1", "cand", "sc", "sn", "sa", "nk", "esc", "from_exp"]
        for country in ([a.country] if a.country else countries_of(a.work, a.split)):
            npool = 0
            for F in iter_features(a.work, a.split, country):
                p = ranker_score(R["model"], F[R["cols"]].values.astype(np.float32))
                # an S1 can never receive more than ~2x the average budget: drop the tail early (memory)
                k = group_rank(F.s1.values, p) < int(2 * a.budget + 5)
                n_s1 += F.s1.nunique()
                npool += len(F)
                parts.append(F.loc[k, keepcols])
                probs.append(p[k])
                del F
                gc.collect()
            log(f"   scored {country}: {npool} pool pairs")
        P = pd.concat(parts, ignore_index=True)
        p = np.concatenate(probs)
        keep, thr = select(P, p, a.budget, n_s1)
        B = P[keep].reset_index(drop=True)
        B["rp"] = p[keep]
        # columns expected by train.py / predict.py
        B = B.rename(columns={"sc": "bscore", "sn": "bname", "sa": "baddr", "nk": "nkeys"})
        B["brank"] = group_rank(B.s1.values, B.rp.values)
        B["rank_n"] = group_rank(B.s1.values, B.bname.values)
        B["rank_a"] = group_rank(B.s1.values, B.baddr.values)
        B["rrank"] = B.brank.values
        save_df(B, f"{a.work}/blocks_{a.split}_v7")
        log(f"saved blocks_{a.split}_v7: {len(B)} pairs, {len(B) / n_s1:.2f}/S1, cutoff {thr:.3f}")
        if a.split == "train":
            from data import gt_int
            gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
            vids, _ = val_ids(a.work)
            if a.eval_present:
                vids = np.intersect1d(vids, B.s1.values)
            vs = B[np.isin(B.s1.values, vids)].groupby("s1").cand.apply(set).to_dict()
            tot = sum(len(gt.get(s, ())) for s in vids)
            r = sum(len(gt.get(s, set()) & vs.get(s, set())) for s in vids) / tot
            log(f"VAL recall {r:.4f}  ceiling {ceiling(gt, vs, vids):.4f}")


if __name__ == "__main__":
    main()
