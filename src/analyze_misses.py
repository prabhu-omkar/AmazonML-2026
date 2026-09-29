"""Blocking diagnostics on the validation S1 entities (same 40k split as train.py):
  1. taxonomy of true matches missed by the current candidate cut (K=12 + 3 name + 3 address)
  2. ceiling macro F0.5 (perfect matcher on the candidate set)
  3. query expansion / second hop: re-query the index with the keys of each S1's top-k candidates
     and measure recall gain per added candidate, versus simply raising K.

python src/analyze_misses.py --work work3 --data dataset --n_val 40000 --workers 2
"""
import argparse

from common import *
from data import load_fields, gt_int
from block import keys_from_parts, part_files, countries_of, record_keys, _CAT
from features import _soundex

M32 = np.uint64(0xFFFFFFFF)


def build_index(ck, nC):
    ch = (ck >> np.uint64(32)).astype(np.uint32)
    cri = (ck & M32).astype(np.int32)
    brk = np.flatnonzero(ch[1:] != ch[:-1]) + 1
    ustart = np.concatenate([[0], brk]).astype(np.int32)
    ucnt = np.diff(np.concatenate([ustart, [len(ch)]])).astype(np.int32)
    uk = ch[ustart]
    del ch, brk
    idf = np.log1p(nC / ucnt).astype(np.float32)
    return dict(uk=uk, ustart=ustart, ucnt=ucnt, idf=idf, cri=cri, nC=nC)


def query(ix, qh, qri, nQ, max_df, top_m, top_side, max_expand=10_000_000):
    """qh: uint32 key hashes, qri: int32 query row. returns DataFrame(q,c,bscore,bname,baddr,brank,rank_n,rank_a)"""
    uk, ustart, ucnt, idf, cri, nC = ix["uk"], ix["ustart"], ix["ucnt"], ix["idf"], ix["cri"], ix["nC"]
    o = np.argsort(qri, kind="stable")
    qri, qh = qri[o], qh[o]
    pos = np.searchsorted(uk, qh)
    pos[pos >= len(uk)] = 0
    hit = uk[pos] == qh
    cnt = np.where(hit, ucnt[pos], 0)
    ok = (cnt > 0) & (cnt <= max_df)
    qri, pos, cnt = qri[ok], pos[ok], cnt[ok]
    start, w = ustart[pos], idf[pos]
    kc = (uk[pos] >> np.uint32(29)).astype(np.int8)
    qstart = np.searchsorted(qri, np.arange(nQ + 1))
    ccum = np.concatenate([[0], np.cumsum(cnt)])
    out = []
    b0 = 0
    while b0 < nQ:
        b1 = int(np.searchsorted(ccum[qstart], ccum[qstart[b0]] + max_expand, side="right")) - 1
        b1 = min(max(b1, b0 + 1), nQ)
        a0, a1 = qstart[b0], qstart[b1]
        b0 = b1
        if a1 <= a0:
            continue
        cc = cnt[a0:a1]
        tot = int(cc.sum())
        rep_q = np.repeat(qri[a0:a1], cc)
        rep_w = np.repeat(w[a0:a1], cc)
        rep_k = np.repeat(kc[a0:a1], cc)
        csum = np.cumsum(cc) - cc
        idx = np.arange(tot, dtype=np.int64) - np.repeat(csum, cc) + np.repeat(start[a0:a1], cc)
        code = rep_q.astype(np.int64) * nC + cri[idx]
        del idx, rep_q
        o = np.argsort(code, kind="stable")
        code, rep_w, rep_k = code[o], rep_w[o], rep_k[o]
        st = np.concatenate([[0], np.flatnonzero(code[1:] != code[:-1]) + 1])
        uc = code[st]
        sc = np.add.reduceat(rep_w, st)
        sn = np.add.reduceat(np.where(rep_k == 0, rep_w, 0), st)
        sa = np.add.reduceat(np.where(rep_k == 1, rep_w, 0), st)
        qq = (uc // nC).astype(np.int32)
        c2 = (uc % nC).astype(np.int32)

        def ranks(score):
            oo = np.lexsort((-score, qq))
            r = np.empty(len(qq), np.int32)
            q_s = qq[oo]
            r[oo] = np.arange(len(qq)) - np.searchsorted(q_s, q_s, side="left")
            return r
        rt, rn, ra = ranks(sc), ranks(sn), ranks(sa)
        keep = (rt < top_m) | ((rn < top_side) & (sn > 0)) | ((ra < top_side) & (sa > 0))
        out.append(pd.DataFrame({"q": qq[keep], "c": c2[keep], "bscore": sc[keep], "bname": sn[keep],
                                 "baddr": sa[keep], "brank": rt[keep], "rank_n": rn[keep], "rank_a": ra[keep]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["q", "c", "bscore", "brank"])


def cut(P, K, side):
    m = P.brank < K
    if side:
        m |= ((P.rank_n < side) & (P.bname > 0)) | ((P.rank_a < side) & (P.baddr > 0))
    return P[m]


def select_rows(packed, look, step=10_000_000):
    """look: int array (n_rows) mapping row -> new id or -1. returns (hash32, new_id) of matching keys"""
    H, R = [], []
    for i in range(0, len(packed), step):
        p = packed[i:i + step]
        r = look[(p & M32).astype(np.int32)]
        sel = r >= 0
        H.append((p[sel] >> np.uint64(32)).astype(np.uint32)); R.append(r[sel].astype(np.int32))
    return np.concatenate(H), np.concatenate(R)


def rows_keys(packed, rows_wanted, n_rows):
    """dict row -> set(hash32) for the wanted rows"""
    look = np.full(n_rows, -1, np.int64)
    w = np.asarray(sorted(rows_wanted), dtype=np.int64)
    look[w] = w
    hh, rr = select_rows(packed, look)
    d = {}
    for a, b in zip(rr.tolist(), hh.tolist()):
        d.setdefault(a, set()).add(b)
    return d


def f05_ceiling(gt, pool_sets, ids):
    tot = 0.0
    for s in ids:
        g = gt.get(s, set())
        if not g:
            tot += 1.0
            continue
        r = len(g & pool_sets.get(s, set())) / len(g)
        tot += 1.25 * r / (0.25 + r) if r > 0 else 0.0
    return tot / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work3")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--n_val", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_df", type=int, default=300)
    ap.add_argument("--out", default="logs/misses")
    ap.add_argument("--country", default="")
    ap.add_argument("--seeds", default="top1,top2,top3,oracle_true")
    a = ap.parse_args()
    gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
    s1_meta = load_fields(a.work, "train", [1], None, None, cols=["entity_id", "country"])
    perm = s1_meta.entity_id.values[np.random.RandomState(42).permutation(len(s1_meta))]
    val = set(perm[:a.n_val].tolist())
    rows_out, summary = [], []
    for country in ([a.country] if a.country else countries_of(a.work, "train")):
        vc = set(s1_meta.entity_id.values[(s1_meta.country.values == country)].tolist()) & val
        cache = f"{a.work}/tmp_ck_{country}.npz"
        if os.path.exists(cache):
            z = np.load(cache); c_ids, ck = z["ids"], z["ck"]
        else:
            c_ids, ck = keys_from_parts(part_files(a.work, "train", 2) + part_files(a.work, "train", 3), country, a.workers)
            np.savez(cache, ids=c_ids, ck=ck)
        from data import int_to_id
        q_ids, qk = keys_from_parts(part_files(a.work, "train", 1), country, a.workers, set(int_to_id(np.array(sorted(vc)))))
        nC, nQ = len(c_ids), len(q_ids)
        log(f"[{country}] val S1={nQ} cands={nC} keys={len(ck)}")
        ix = build_index(ck, nC)
        crow_of = pd.Series(np.arange(nC), index=c_ids)
        qh = (qk >> np.uint64(32)).astype(np.uint32)
        qr = (qk & M32).astype(np.int32)
        R = query(ix, qh, qr, nQ, a.max_df, 60, 3)
        R["s1"] = q_ids[R.q.values]
        R["cand"] = c_ids[R.c.values]
        log(f"   full ranking pairs {len(R)}")
        ids = q_ids.tolist()
        G = [(s, m) for s in ids for m in gt.get(s, ())]
        tot = len(G)
        pool = cut(R, 12, 3)
        pool_sets = pool.groupby("s1").cand.apply(set).to_dict()
        R_rank = {(s, c): r for s, c, r in zip(R.s1.values, R.cand.values, R.brank.values)}
        miss = [(s, m) for s, m in G if m not in pool_sets.get(s, ())]
        base_rec = 1 - len(miss) / tot
        base_ceil = f05_ceiling(gt, pool_sets, ids)
        log(f"   base: recall {base_rec:.4f} avg cands {len(pool) / nQ:.2f} ceiling F0.5 {base_ceil:.4f}")
        # ---- taxonomy
        q_row = pd.Series(np.arange(nQ), index=q_ids)
        need_c = [crow_of.get(m, -1) for _, m in miss]
        miss_rows_c = {r for r in need_c if r >= 0}
        miss_rows_q = {q_row[s] for s, _ in miss}
        ckeys = rows_keys(ck, miss_rows_c, nC) if miss_rows_c else {}
        qkeys = rows_keys(qk, miss_rows_q, nQ) if miss_rows_q else {}
        # text fields for misses (address empty, name ambiguity, phonetic)
        s1f = load_fields(a.work, "train", [1], None, country, cols=["entity_id", "country", "core", "ad"])
        core_cnt = s1f.core.value_counts()
        s1f = s1f.set_index("entity_id")
        cf = load_fields(a.work, "train", [2, 3], {m for _, m in miss}, country,
                         cols=["entity_id", "country", "core", "ad"]).set_index("entity_id")
        uk, ucnt = ix["uk"], ix["ucnt"]
        for (s, m), cr in zip(miss, need_c):
            rk = R_rank.get((s, m))
            rec = {"country": country, "s1": s, "m": m}
            if cr < 0:
                cat = "not_in_country"
            elif rk is not None:
                cat = "rank_13_45" if rk < 45 else "rank_46_60"
            else:
                sh = qkeys.get(q_row[s], set()) & ckeys.get(cr, set())
                if not sh:
                    cat = "no_shared_key"
                else:
                    hs = np.array(sorted(sh), dtype=np.uint32)
                    p = np.searchsorted(uk, hs)
                    p[p >= len(uk)] = 0
                    dfs = np.where(uk[p] == hs, ucnt[p], 0)
                    cat = "only_df_gt_cap" if (dfs > a.max_df).all() else "rank_gt_60"
                    rec["min_df"] = int(dfs.min())
            rec["cat"] = cat
            r1 = s1f.loc[s] if s in s1f.index else None
            r2 = cf.loc[m] if m in cf.index else None
            if r1 is not None and r2 is not None:
                rec["ad_empty2"] = int(len(r2.ad) == 0)
                rec["ad_empty1"] = int(len(r1.ad) == 0)
                rec["s1_core_dup"] = int(core_cnt.get(r1.core, 0))
                rec["core_eq"] = int(r1.core == r2.core)
                t1 = {_soundex(t) for t in r1.core.split() if t}
                t2 = {_soundex(t) for t in r2.core.split() if t}
                rec["sdx_share"] = len(t1 & t2)
                rec["sdx_all"] = int(bool(t1) and t1 == t2)
                rec["s1_core"], rec["m_core"] = r1.core, r2.core
                rec["s1_ad"], rec["m_ad"] = r1.ad[:60], r2.ad[:60]
            rows_out.append(rec)
        # ---- raising K (same budget comparison)
        for K in [14, 16, 20, 30, 45]:
            pk = cut(R, K, 3)
            ps = pk.groupby("s1").cand.apply(set).to_dict()
            rec_k = sum(m in ps.get(s, ()) for s, m in G) / tot
            summary.append((country, f"K={K}+3+3", rec_k, len(pk) / nQ, f05_ceiling(gt, ps, ids)))
        summary.append((country, "BASE K=12+3+3", base_rec, len(pool) / nQ, base_ceil))
        # ---- query expansion / second hop: seeds = top-k pool candidates by blocking score
        pool = pool.sort_values(["s1", "bscore"], ascending=[True, False])
        pool["r"] = pool.groupby("s1").cumcount()
        seeds_all = {"top1": pool[pool.r < 1], "top2": pool[pool.r < 2], "top3": pool[pool.r < 3]}
        # oracle: true matches that are already in the pool (upper bound of a perfect 2nd hop)
        pool["y"] = [c in gt.get(s, ()) for s, c in zip(pool.s1.values, pool.cand.values)]
        seeds_all["oracle_true"] = pool[pool.y]
        for name, S in seeds_all.items():
            if name not in a.seeds.split(","):
                continue
            import gc; gc.collect()
            urows = np.unique(crow_of.reindex(S.cand.values).values.astype(np.int64))
            look = np.full(nC, -1, np.int64)
            look[urows] = np.arange(len(urows))
            eh, er = select_rows(ck, look)
            E = query(ix, eh, er, len(urows), a.max_df, 8, 0)
            E["seed"] = c_ids[urows[E.q.values]]
            E["cand"] = c_ids[E.c.values]
            E = E[E.seed != E.cand]
            J = S[["s1", "cand"]].rename(columns={"cand": "seed"}).merge(E[["seed", "cand", "bscore"]], on="seed")
            J = J.groupby(["s1", "cand"], as_index=False).bscore.max()
            inpool = set(zip(pool.s1.values, pool.cand.values))
            J = J[[(s, c) not in inpool for s, c in zip(J.s1.values, J.cand.values)]]
            J = J.sort_values(["s1", "bscore"], ascending=[True, False])
            J["r"] = J.groupby("s1").cumcount()
            for N in [1, 2, 3, 5, 8]:
                add = J[J.r < N]
                ps = {k: set(v) for k, v in pool_sets.items()}
                for s, c in zip(add.s1.values, add.cand.values):
                    ps.setdefault(s, set()).add(c)
                rec_n = sum(m in ps.get(s, ()) for s, m in G) / tot
                summary.append((country, f"expand {name} +{N}", rec_n, (len(pool) + len(add)) / nQ,
                                f05_ceiling(gt, ps, ids)))
            log(f"   expansion {name} done")
            del E, J, eh, er, look
        del ix, ck, qk, R
        import gc; gc.collect()
    D = pd.DataFrame(rows_out)
    a.out += ("_" + a.country) if a.country else ""
    D.to_csv(a.out + "_pairs.tsv", sep="\t", index=False)
    S = pd.DataFrame(summary, columns=["country", "setting", "recall", "avg_cands", "ceiling_f05"])
    S.to_csv(a.out + "_summary.tsv", sep="\t", index=False)
    print(S.to_string())
    print("\nmiss taxonomy (share of missed true pairs):")
    print(D.groupby(["country", "cat"]).size().unstack(0).fillna(0).astype(int))
    if len(D):
        print("\nmisses: address-empty (cand) share", round(D.ad_empty2.mean(), 3),
              " S1 core shared by >=5 S1:", round((D.s1_core_dup >= 5).mean(), 3),
              " exact core equal:", round(D.core_eq.mean(), 3))
        print(D.groupby("cat")[["ad_empty2", "core_eq", "sdx_all"]].mean().round(3))


if __name__ == "__main__":
    main()
