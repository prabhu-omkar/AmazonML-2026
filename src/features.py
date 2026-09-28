"""Step 3: pair features for (S1, candidate) pairs, computed per country.

Vectorised: hashed char-n-gram / word TF-IDF cosines via scipy sparse (records vectorised once,
pairs gathered by row index) + python set features in a process pool + blocking-graph context.
"""
from multiprocessing import Pool

import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from common import *

NF = 2 ** 20


def _hv(kind):
    if kind == "w":
        return HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=NF,
                                 alternate_sign=False, norm=None, lowercase=False)
    n = int(kind[1])
    return HashingVectorizer(analyzer="char_wb", ngram_range=(n, n), n_features=NF,
                             alternate_sign=False, norm=None, lowercase=False)


def _transform(args):
    kind, texts = args
    X = _hv(kind).transform(texts).astype(np.float32)
    X.data = np.minimum(X.data, 2.0)
    return X


def vectorise(texts, kind, pool, idf=None, step=100000):
    parts = pool.map(_transform, [(kind, texts[i:i + step]) for i in range(0, len(texts), step)])
    X = sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, NF), dtype=np.float32)
    if idf is not None:
        X = X @ sp.diags(idf)
    return normalize(X, norm="l2", copy=False).tocsr()


def fit_idf(X_raw_bin_sum, n):
    return (np.log((1 + n) / (1 + X_raw_bin_sum)) + 1).astype(np.float32)


def rowdot_idx(A, ia, B, ib, step=2000000):
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), step):
        a = A[ia[i:i + step]]
        b = B[ib[i:i + step]]
        out[i:i + step] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


LEGAL = LEGAL_FORMS
SET_COLS = ["core_jac", "core_ovl_min", "concat_eq", "concat_sub", "first_tok_eq", "legal_conflict",
            "legal_common", "n_core1", "n_core2", "num_common", "num_jac", "first_num_eq", "zip_eq",
            "zip_conflict", "alph_jac", "addr_len1", "addr_len2", "num_conflict_first",
            "first_num_sim", "best_num_sim", "tok_fuzzy_cov1", "tok_fuzzy_cov2", "alph_fuzzy_cov1",
            "name_in_other", "n_nums1", "n_nums2",
            "nm_sh_idf", "nm_sh_idf_max", "nm_un1_idf", "nm_un2_idf", "ad_sh_idf", "ad_un1_idf", "ad_un2_idf",
            "ad_cov2", "num_cov2", "nm_digit_tok2", "core_cov2", "core_cov1", "last_alph_eq",
            "x2_sib_max", "x2_sib_mean", "x2_n_sib", "x2_n_noise", "x2_n_unk", "x1_sib_max", "x1_n_sib", "x1_n_noise",
            "x2_agree_min", "x2_n_lowagree", "x1_agree_min", "x1_n_lowagree",
            "lev_nm", "jw_core", "tsr_nm", "lev_ad", "tsr_ad", "pr_nm", "jw_first", "lev_street",
            "sdx_cov1", "sdx_cov2"]

try:  # fast C++ edit distances (pip install rapidfuzz); features are -1 when unavailable
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz.distance import JaroWinkler as _rf_jw, Levenshtein as _rf_lev
    _HAS_RF = True
except Exception:  # pragma: no cover
    _HAS_RF = False

_TOK = {"core": {}, "alph": {}, "n": 1.0, "xtok": {}, "agree": {}}


_SDX = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")


def _soundex(t):
    """phonetic key robust to transliteration variants (laxmi ~ lakshmi, sri ~ shri)"""
    if not t:
        return ""
    if t[0].isdigit():
        return t
    head = t[0]
    code = t[1:].translate(_SDX)
    out, prev = [], head.translate(_SDX)
    for ch in code:
        if ch.isdigit() and ch != prev:
            out.append(ch)
        if ch not in "hw":
            prev = ch
    return (head + "".join(out) + "000")[:4]


_STREET_TYPES = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "cir", "pl", "hwy", "pkwy", "ter", "trl", "sq",
                 "rue", "rte", "ch", "imp", "all", "way", "marg", "ngr", "col"}


def _street(alph):
    """street name = alphabetic tokens up to and including the first street-type word"""
    toks = alph.split()
    for j, t in enumerate(toks[:8]):
        if t in _STREET_TYPES and j > 0:
            return " ".join(toks[max(0, j - 3):j + 1])
    return " ".join(toks[:2])


def _init_tok(core_df, alph_df, n, xtok=None, agree=None):
    _TOK["core"], _TOK["alph"], _TOK["n"] = core_df, alph_df, float(n)
    _TOK["xtok"] = xtok or {}
    _TOK["agree"] = agree or {}


def _agree_stats(tokens):
    ag = _TOK["agree"]
    vals = [ag[t] for t in tokens if t in ag]
    if not vals:
        return -1.0, 0
    return min(vals), sum(1 for v in vals if v < 0.3)


def _xstats(tokens):
    """tokens present in one name but not the other -> (max neg-rate, mean, #sibling-words, #noise-words, #unknown)"""
    xt = _TOK["xtok"]
    if not tokens:
        return 0.0, 0.0, 0, 0, 0
    rates, ns, nn, nu = [], 0, 0, 0
    for t in tokens:
        r = xt.get(t)
        if r is None:
            nu += 1
            continue
        rates.append(r)
        if r >= 0.85:
            ns += 1
        elif r <= 0.4:
            nn += 1
    if not rates:
        return -1.0, -1.0, 0, 0, nu
    return max(rates), sum(rates) / len(rates), ns, nn, nu


def _idf(t, table):
    return math.log((_TOK["n"] + 1.0) / (table.get(t, 0) + 1.0))


def _num_sim(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == lb:
        d = sum(x != y for x, y in zip(a, b))
        return 0.8 if d == 1 else (0.4 if d == 2 and la >= 4 else 0.0)
    if min(la, lb) >= 2 and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)):
        return 0.7 if abs(la - lb) == 1 else 0.5
    return 0.0


def _tok_close(t, others):
    if t in others:
        return 1.0
    for o in others:
        if len(t) >= 4 and len(o) >= 4 and (t[:3] == o[:3] or t[-3:] == o[-3:]) and abs(len(t) - len(o)) <= 2:
            return 0.7
    return 0.0


def _set_feats(args):
    (n1, n2, c1, c2, nu1, nu2, a1, a2, ad1, ad2) = args
    out = np.zeros((len(n1), len(SET_COLS)), dtype=np.float32)
    for i in range(len(n1)):
        t1, t2 = n1[i].split(), n2[i].split()
        k1, k2 = c1[i].split(), c2[i].split()
        s1, s2 = set(k1), set(k2)
        inter = len(s1 & s2)
        uni = len(s1 | s2) or 1
        out[i, 0] = inter / uni
        out[i, 1] = inter / (min(len(s1), len(s2)) or 1)
        j1, j2 = "".join(k1), "".join(k2)
        out[i, 2] = 1.0 if (j1 and j1 == j2) else 0.0
        out[i, 3] = 1.0 if (j1 and j2 and (j1 in j2 or j2 in j1)) else 0.0
        out[i, 4] = 1.0 if (k1 and k2 and k1[0] == k2[0]) else 0.0
        l1 = set(t for t in t1 if t in LEGAL)
        l2 = set(t for t in t2 if t in LEGAL)
        out[i, 5] = 1.0 if (l1 and l2 and not (l1 & l2)) else 0.0
        out[i, 6] = len(l1 & l2)
        out[i, 7] = len(k1)
        out[i, 8] = len(k2)
        m1, m2 = nu1[i].split(), nu2[i].split()
        q1, q2 = set(m1), set(m2)
        out[i, 9] = len(q1 & q2)
        out[i, 10] = len(q1 & q2) / len(q1 | q2) if (q1 and q2) else -1
        out[i, 11] = (1.0 if m1[0] == m2[0] else 0.0) if (m1 and m2) else -1
        z1 = set(x for x in m1 if len(x) in (5, 6))
        z2 = set(x for x in m2 if len(x) in (5, 6))
        out[i, 12] = (1.0 if (z1 & z2) else 0.0) if (z1 and z2) else -1
        out[i, 13] = 1.0 if (z1 and z2 and not (z1 & z2)) else 0.0
        b1, b2 = set(a1[i].split()), set(a2[i].split())
        out[i, 14] = len(b1 & b2) / len(b1 | b2) if (b1 and b2) else -1
        out[i, 15] = len(b1) + len(q1)
        out[i, 16] = len(b2) + len(q2)
        out[i, 17] = 1.0 if (m1 and m2 and m1[0] not in q2) else 0.0
        if m1 and m2:
            out[i, 18] = _num_sim(m1[0], m2[0])
            out[i, 19] = max(_num_sim(x, y) for x in m1[:4] for y in m2[:4])
        else:
            out[i, 18] = out[i, 19] = -1
        if k1 and k2:
            out[i, 20] = sum(_tok_close(t, s2) for t in s1) / len(s1)
            out[i, 21] = sum(_tok_close(t, s1) for t in s2) / len(s2)
        if b1 and b2:
            bb1 = [t for t in b1 if len(t) >= 3]
            out[i, 22] = (sum(_tok_close(t, b2) for t in bb1) / len(bb1)) if bb1 else -1
        else:
            out[i, 22] = -1
        out[i, 23] = 1.0 if (j1 and j2 and (j1 in "".join(t2) or j2 in "".join(t1))) else 0.0
        out[i, 24] = len(m1)
        out[i, 25] = len(m2)
        tc, ta = _TOK["core"], _TOK["alph"]
        sh = [_idf(t, tc) for t in (s1 & s2)]
        out[i, 26] = sum(sh)
        out[i, 27] = max(sh) if sh else 0.0
        out[i, 28] = sum(_idf(t, tc) for t in (s1 - s2))
        out[i, 29] = sum(_idf(t, tc) for t in (s2 - s1))
        out[i, 30] = sum(_idf(t, ta) for t in (b1 & b2))
        out[i, 31] = sum(_idf(t, ta) for t in (b1 - b2))
        out[i, 32] = sum(_idf(t, ta) for t in (b2 - b1))
        out[i, 33] = (len(b1 & b2) / len(b2)) if b2 else -1
        out[i, 34] = (len(q1 & q2) / len(q2)) if q2 else -1
        out[i, 35] = sum(1 for t in t2 if t.isdigit() and t not in t1)
        out[i, 36] = (inter / len(s2)) if s2 else 0.0
        out[i, 37] = (inter / len(s1)) if s1 else 0.0
        la1, la2 = a1[i].split(), a2[i].split()
        out[i, 38] = (1.0 if set(la1[-2:]) & set(la2[-2:]) else 0.0) if (la1 and la2) else -1
        mx, mn, ns, nn, nu = _xstats(s2 - s1)
        out[i, 39], out[i, 40], out[i, 41], out[i, 42], out[i, 43] = mx, mn, ns, nn, nu
        mx, mn, ns, nn, nu = _xstats(s1 - s2)
        out[i, 44], out[i, 45], out[i, 46] = mx, ns, nn
        out[i, 47], out[i, 48] = _agree_stats(s2 - s1)
        out[i, 49], out[i, 50] = _agree_stats(s1 - s2)
        if _HAS_RF:
            out[i, 51] = _rf_lev.normalized_similarity(n1[i], n2[i])
            out[i, 52] = _rf_jw.normalized_similarity(j1, j2) if (j1 and j2) else 0.0
            out[i, 53] = _rf_fuzz.token_set_ratio(n1[i], n2[i]) / 100.0
            if ad1[i] and ad2[i]:
                out[i, 54] = _rf_lev.normalized_similarity(ad1[i], ad2[i])
                out[i, 55] = _rf_fuzz.token_set_ratio(ad1[i], ad2[i]) / 100.0
            else:
                out[i, 54] = out[i, 55] = -1
            out[i, 56] = _rf_fuzz.partial_ratio(j1, j2) / 100.0 if (j1 and j2) else 0.0
            out[i, 57] = _rf_jw.normalized_similarity(k1[0], k2[0]) if (k1 and k2) else 0.0
            st1, st2 = _street(a1[i]), _street(a2[i])
            out[i, 58] = _rf_lev.normalized_similarity(st1, st2) if (st1 and st2) else -1
        else:
            out[i, 51:59] = -1
        if s1 and s2:
            x1 = set(_soundex(t) for t in s1)
            x2 = set(_soundex(t) for t in s2)
            out[i, 59] = len(x1 & x2) / len(x1)
            out[i, 60] = len(x1 & x2) / len(x2)
    return out


VEC_SPECS = [("nm", "c3"), ("nm", "c2"), ("nm", "c4"), ("core", "c3"), ("core", "w"),
             ("ad", "c3"), ("ad", "w"), ("alph", "c4")]


def _tokdf(texts):
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(set(t.split()))
    return dict(c)


def name_key_hash(core_series):
    return pd.util.hash_array(np.array(["".join(x.split()) for x in core_series], dtype=object), categorize=False)


def addr_key_hash(nums, alph):
    keys = []
    for n, a in zip(nums, alph):
        nn = n.split()[:1]
        aa = [t for t in a.split() if len(t) >= 3][:2]
        keys.append(" ".join(nn + aa) if (nn and aa) else "")
    return pd.util.hash_array(np.array(keys, dtype=object), categorize=False), np.array([k == "" for k in keys])


def fit_idfs(cf, workers, idf_sample=400000):
    rng = np.random.RandomState(0)
    samp = rng.choice(len(cf), size=min(idf_sample, len(cf)), replace=False)
    idfs = {"_tok": (_tokdf(cf.core.values[samp]), _tokdf(cf.alph.values[samp]), len(samp))}
    with Pool(workers) as pool:
        for col, kind in VEC_SPECS:
            t = cf[col].values[samp].tolist()
            Xs = vectorise(t, kind, pool)
            Xs.data[:] = 1
            idfs[(col, kind)] = fit_idf(np.asarray(Xs.sum(axis=0)).ravel(), Xs.shape[0])
    return idfs


def base_features(pairs, s1f, cf, workers, idfs):
    """pair-level features (no group context). s1f/cf may be supersets; only needed rows vectorised."""
    u1, i1 = np.unique(pd.Index(s1f.entity_id).get_indexer(pairs.s1.values), return_inverse=True)
    u2, i2 = np.unique(pd.Index(cf.entity_id).get_indexer(pairs.cand.values), return_inverse=True)
    assert (u1 >= 0).all() and (u2 >= 0).all()
    s1f = s1f.iloc[u1].reset_index(drop=True)
    cf = cf.iloc[u2].reset_index(drop=True)
    F = {}
    tok = tuple(idfs.get("_tok", ({}, {}, 1))) + (idfs.get("_xtok", {}), idfs.get("_agree", {}))
    with Pool(workers, initializer=_init_tok, initargs=tok) as pool:
        for col, kind in VEC_SPECS:
            A = vectorise(s1f[col].tolist(), kind, pool, idfs[(col, kind)])
            B = vectorise(cf[col].tolist(), kind, pool, idfs[(col, kind)])
            F[f"cos_{kind}_{col}"] = rowdot_idx(A, i1, B, i2)
            del A, B
        step = 50000
        tasks = []
        cols = ["nm", "core", "nums", "alph", "ad"]
        S1v = {c: s1f[c].values for c in cols}
        CV = {c: cf[c].values for c in cols}
        for i in range(0, len(pairs), step):
            a, b = i1[i:i + step], i2[i:i + step]
            tasks.append((S1v["nm"][a].tolist(), CV["nm"][b].tolist(), S1v["core"][a].tolist(), CV["core"][b].tolist(),
                          S1v["nums"][a].tolist(), CV["nums"][b].tolist(), S1v["alph"][a].tolist(), CV["alph"][b].tolist(),
                          S1v["ad"][a].tolist(), CV["ad"][b].tolist()))
        res = pool.map(_set_feats, tasks, chunksize=1)
    S = np.vstack(res) if res else np.zeros((0, len(SET_COLS)), np.float32)
    for j, c in enumerate(SET_COLS):
        F[c] = S[:, j]
    del S, res, tasks
    ad_len2 = cf.ad.str.len().values[i2]
    F["ad_empty2"] = (ad_len2 == 0).astype(np.float32)
    F["is_dom2"] = cf.is_dom.values[i2].astype(np.float32)
    F["has_dba2"] = cf.has_dba.values[i2].astype(np.float32)
    F["src"] = (np.asarray(pairs.cand.values) // 10 ** 10 == 3).astype(np.float32)
    l1 = s1f.nm.str.len().values[i1].astype(np.float32)
    l2 = cf.nm.str.len().values[i2].astype(np.float32)
    F["len_ratio_nm"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    F["name_x_addr"] = F["cos_c3_nm"] * np.where(F["ad_empty2"] == 1, 0.5, F["cos_c3_ad"])
    rar = idfs.get("_rar")
    if rar is not None:
        h1 = name_key_hash(s1f.core.values)
        h2 = name_key_hash(cf.core.values)
        F["nmfreq_s1_in_s1"] = rar["nm_s1"].reindex(h1).fillna(0).values[i1].astype(np.float32)
        F["nmfreq_s1_in_c"] = rar["nm_c"].reindex(h1).fillna(0).values[i1].astype(np.float32)
        F["nmfreq_c_in_c"] = rar["nm_c"].reindex(h2).fillna(0).values[i2].astype(np.float32)
        F["nmfreq_c_in_s1"] = rar["nm_s1"].reindex(h2).fillna(0).values[i2].astype(np.float32)
        a1, e1 = addr_key_hash(s1f.nums.values, s1f.alph.values)
        a2, e2 = addr_key_hash(cf.nums.values, cf.alph.values)
        v1 = rar["ad_c"].reindex(a1).fillna(0).values
        v2 = rar["ad_c"].reindex(a2).fillna(0).values
        F["adfreq_s1_in_c"] = np.where(e1, -1, v1)[i1].astype(np.float32)
        F["adfreq_c_in_c"] = np.where(e2, -1, v2)[i2].astype(np.float32)
        F["adkey_eq"] = np.where(e1[i1] | e2[i2], -1, (a1[i1] == a2[i2]).astype(np.float32)).astype(np.float32)
    return pd.DataFrame(F)


def featurize_country(pairs, s1f, cf, workers, idfs=None):
    if idfs is None:
        idfs = fit_idfs(cf, workers)
    F = base_features(pairs, s1f, cf, workers, idfs)
    return context_features(pairs, F)


def context_features(p, F):
    s1v = p.s1.values
    cv = p.cand.values
    F["bscore"] = p.bscore.values.astype(np.float32)
    F["nkeys"] = p.nkeys.values.astype(np.float32)
    F["brank"] = p.brank.values.astype(np.float32)
    for c in ["bname", "baddr", "rank_n", "rank_a", "rp", "esc", "from_exp", "rrank"]:  # rp..: v7 ranker blocks
        if c in p.columns:
            F[c] = p[c].values.astype(np.float32)
    bs = pd.Series(p.bscore.values)
    F["b_rel"] = (bs / bs.groupby(s1v).transform("max")).values.astype(np.float32)
    F["n_cand_s1"] = bs.groupby(s1v).transform("size").values.astype(np.float32)
    F["n_s1_for_cand"] = p.cand_nS1.values.astype(np.float32)
    F["cand_rank_among_s1"] = p.cand_rank.values.astype(np.float32)
    F["cand_gap_best_other"] = p.cand_gap.values.astype(np.float32)
    for c in ["cos_c3_nm", "cos_c3_ad", "name_x_addr", "cos_w_core"]:
        v = pd.Series(F[c].values)
        F[c + "_rel1"] = (v - v.groupby(s1v).transform("max")).values.astype(np.float32)
        F[c + "_relc"] = (v - v.groupby(cv).transform("max")).values.astype(np.float32)
        F[c + "_rk1"] = v.groupby(s1v).rank(ascending=False, method="min").values.astype(np.float32)
    return F


def competition_columns(pairs):
    """cand-side competition from blocking scores (needs ALL S1 of the split)."""
    g = pairs.groupby("cand", sort=False).bscore
    pairs["cand_nS1"] = g.transform("size").astype(np.int16)
    pairs["cand_rank"] = g.rank(ascending=False, method="min").astype(np.int16)
    mx = g.transform("max")
    # second max per cand
    srt = pairs[["cand", "bscore"]].sort_values(["cand", "bscore"], ascending=[True, False])
    second = srt.groupby("cand").bscore.nth(1)
    second = pd.Series(second.values, index=srt.loc[second.index, "cand"].values)
    sec = pairs.cand.map(second).fillna(0).values
    best_other = np.where(pairs.bscore.values >= mx.values, sec, mx.values)
    pairs["cand_gap"] = (pairs.bscore.values - best_other).astype(np.float32)
    return pairs


def learn_extra_token_table(pairs_s1_core, pairs_c_core, labels, min_support=30, prior=0.5, k=20):
    """For candidate pairs whose names overlap strongly, learn P(non-match | token appears in only one
    of the two names).  Generated look-alike 'sibling' businesses add words such as 'group', 'exports',
    'ventures', whereas noise on genuine matches adds 'services', 'dba', 'center'...  Smoothed rates."""
    from collections import Counter
    pos, neg = Counter(), Counter()
    for a, b, y in zip(pairs_s1_core, pairs_c_core, labels):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            continue
        inter = len(sa & sb)
        if inter / min(len(sa), len(sb)) < 0.99:
            continue
        for t in sa ^ sb:
            (pos if y else neg)[t] += 1
    out = {}
    for t in set(pos) | set(neg):
        n = pos[t] + neg[t]
        if n >= min_support:
            out[t] = float((neg[t] + prior * k) / (n + k))
    return out


def learn_agree_table(s1_core, s1_nums, c_core, c_nums, min_support=15):
    """UNSUPERVISED: for strongly overlapping names, how often does the candidate's house number equal the
    S1's when a given token appears in only one of the names.  Words that create look-alike sibling
    businesses ('group', 'holding', 'groupe', 'participations', ...) almost never co-occur with the same
    address; noise words on genuine matches ('dba', 'services', ...) usually do.  Computed per split and
    per country from that split's own candidate pairs (no labels), so it also works for unseen countries."""
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for a, na, b, nb in zip(s1_core, s1_nums, c_core, c_nums):
        if not na or not nb:
            continue
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb or len(sa & sb) / min(len(sa), len(sb)) < 0.99:
            continue
        eq = na.split()[0] == nb.split()[0]
        for t in sa ^ sb:
            r = agg[t]
            r[0] += eq
            r[1] += 1
    return {t: v[0] / v[1] for t, v in agg.items() if v[1] >= min_support}
