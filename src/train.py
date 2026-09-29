"""Step 3: build train/validation pairs from the train blocking output, featurise,
train the GBDT matcher, tune the decision threshold for macro F0.5 on held-out S1 entities.

python src/train.py --work work --data dataset
"""
import argparse
import os
import pickle

from common import *
from data import load_fields, corpus_sample, gt_int
from features import fit_idfs, base_features, context_features, competition_columns, learn_extra_token_table, learn_agree_table


def cut_candidates(blocks, K, side):
    m = (blocks.brank < K)
    if side > 0:
        m |= ((blocks.rank_n < side) & (blocks.bname > 0)) | ((blocks.rank_a < side) & (blocks.baddr > 0))
    b = blocks[m].reset_index(drop=True)
    # recompute rank within kept set (total score)
    b["brank"] = b.groupby("s1").bscore.rank(ascending=False, method="first").astype(np.int16) - 1
    return b


def rarity_tables(work, split, country, s1_keep=None):
    from features import name_key_hash, addr_key_hash
    s1 = load_fields(work, split, [1], s1_keep, country, cols=["entity_id", "country", "core"])
    nm_s1 = pd.Series(name_key_hash(s1.core.values)).value_counts()
    del s1
    c = load_fields(work, split, [2, 3], None, country, cols=["entity_id", "country", "core", "nums", "alph"])
    nm_c = pd.Series(name_key_hash(c.core.values)).value_counts()
    ah, empty = addr_key_hash(c.nums.values, c.alph.values)
    ad_c = pd.Series(ah[~empty]).value_counts()
    return {"nm_s1": nm_s1, "nm_c": nm_c, "ad_c": ad_c}


def get_idfs(work, split, country, workers, s1_keep=None, tag=""):
    idfs = _get_idfs(work, split, country, workers, s1_keep, tag)
    xp = f"{work}/extra_tok.pkl"
    idfs["_xtok"] = pickle.load(open(xp, "rb")) if os.path.exists(xp) else {}
    ap_ = f"{work}/agree_{split}_{country}{tag}.pkl"
    idfs["_agree"] = pickle.load(open(ap_, "rb")) if os.path.exists(ap_) else {}
    return idfs


def build_agree_table(work, split, country, pairs, n_s1=300000, tag=""):
    """pairs: candidate pairs of this split (any subset); cached per split/country."""
    path = f"{work}/agree_{split}_{country}{tag}.pkl"
    if os.path.exists(path):
        return
    P = pairs
    s1u = P.s1.unique()
    if len(s1u) > n_s1:
        s1u = np.random.RandomState(0).choice(s1u, n_s1, replace=False)
        P = P[P.s1.isin(s1u)]
    c1 = load_fields(work, split, [1], set(P.s1.unique()), country, cols=["entity_id", "country", "core", "nums"])
    c2 = load_fields(work, split, [2, 3], set(P.cand.unique()), country, cols=["entity_id", "country", "core", "nums"])
    m1 = c1.set_index("entity_id"); m2 = c2.set_index("entity_id")
    a = m1.reindex(P.s1.values); b = m2.reindex(P.cand.values)
    tab = learn_agree_table(a.core.fillna("").values, a.nums.fillna("").values,
                            b.core.fillna("").values, b.nums.fillna("").values)
    pickle.dump(tab, open(path, "wb"))
    log(f"  address-agreement table {split}/{country}: {len(tab)} tokens")


def _get_idfs(work, split, country, workers, s1_keep=None, tag=""):
    path = f"{work}/idf3_{split}_{country}{tag}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    idfs = fit_idfs(corpus_sample(work, split, country), workers)
    idfs["_rar"] = rarity_tables(work, split, country, s1_keep)
    pickle.dump(idfs, open(path, "wb"))
    return idfs


def postprocess(s1, cand, prob, thr, one_to_one=True):
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": prob})
    d = d[d.p >= thr]
    if one_to_one and len(d):
        d = d.sort_values("p", ascending=False).drop_duplicates("cand")
    out = {}
    for s, c in zip(d.s1.values, d.cand.values):
        out.setdefault(s, []).append(c)
    return out


def macro_f05_int(pred, gt, ids):
    tot = 0.0
    for i in ids:
        tot += f05_entity(pred.get(i, ()), gt.get(i, ()))
    return tot / len(ids)


def tune_threshold(s1, cand, prob, gt, eval_ids, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    rows = []
    for t in grid:
        f = macro_f05_int(postprocess(s1, cand, prob, t), gt, eval_ids)
        rows.append((float(t), f))
    best = max(rows, key=lambda r: r[1])
    return best, rows


def featurize_pairs(P, work, split, workers, s1_keep=None, tag="", agree_src=None):
    """P: pairs with s1,cand (int ids), blocking cols, competition cols, 'country'."""
    Fs, idxs = [], []
    for country, Pc in P.groupby("country"):
        if agree_src is not None:
            build_agree_table(work, split, country, agree_src[agree_src.country == country], tag=tag)
        idfs = get_idfs(work, split, country, workers, s1_keep, tag)
        s1f = load_fields(work, split, [1], set(Pc.s1.unique()), country)
        cf = load_fields(work, split, [2, 3], set(Pc.cand.unique()), country)
        log(f"  featurising {split}/{country}: {len(Pc)} pairs")
        F = base_features(Pc, s1f, cf, workers, idfs)
        F = context_features(Pc.reset_index(drop=True), F)
        F.index = Pc.index
        Fs.append(F)
    return pd.concat(Fs).loc[P.index]


def make_model(iters, workers):
    try:
        import lightgbm as lgb
        return "lgb", lgb.LGBMClassifier(n_estimators=iters, learning_rate=0.05, num_leaves=255,
                                         min_child_samples=40, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, reg_lambda=1.0, n_jobs=workers, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hgb", HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.08, max_leaf_nodes=127,
                                                     min_samples_leaf=50, l2_regularization=1.0,
                                                     early_stopping=True, validation_fraction=0.05,
                                                     n_iter_no_change=30, random_state=0)


def make_reg_model(workers):
    """strongly regularised GBDT: transfers better to countries unseen in training."""
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=300,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                                  n_jobs=workers, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=300, l2_regularization=5.0, random_state=0)


def train_unseen_country_model(P, F, Pv, Fv, cols, gt, a):
    """Leave-one-country-out: estimate the decision threshold appropriate for a country never seen in
    training (e.g. France), then fit the regularised model on all training countries."""
    countries = sorted(set(P.country.dropna().unique()))
    if len(countries) < 2:
        return None
    thrs = []
    seed = P.is_seed.values
    for held in countries:
        tr = seed & (P.country.values != held)
        va = (Pv.country.values == held) & Pv.is_seed.values
        if tr.sum() == 0 or va.sum() == 0:
            continue
        mdl = make_reg_model(a.workers)
        mdl.fit(F.loc[tr, cols].values, P.y.values[tr])
        pv = mdl.predict_proba(Fv.loc[va, cols].values)[:, 1]
        ids = list(Pv.s1.values[va & Pv.is_seed.values])
        ids = list(dict.fromkeys(ids))
        (t, f), _ = tune_threshold(Pv.s1.values[va], Pv.cand.values[va], pv, gt, ids,
                                   grid=np.round(np.arange(0.5, 0.99, 0.025), 3))
        log(f"  LOCO held-out={held}: F0.5={f:.5f} thr={t}")
        thrs.append(t)
    mdl = make_reg_model(a.workers)
    mdl.fit(F.loc[seed, cols].values, P.y.values[seed])
    thr = float(np.mean(thrs)) if thrs else 0.8
    log(f"  unseen-country model: thr={thr:.3f}")
    return {"model": mdl, "thr": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--side", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=700000)
    ap.add_argument("--n_val", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--blocks", default="blocks_train")
    ap.add_argument("--save_train_feats", type=int, default=0)
    ap.add_argument("--no_closure", type=int, default=1)
    ap.add_argument("--n_models", type=int, default=1)
    ap.add_argument("--train_country", default="")
    ap.add_argument("--val_country", default="")
    ap.add_argument("--drop_feats", default="")
    ap.add_argument("--loco", type=int, default=1)
    ap.add_argument("--n_table", type=int, default=300000)
    a = ap.parse_args()

    gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
    blocks = load_df(f"{a.work}/{a.blocks}")
    pairs = cut_candidates(blocks, a.K, a.side)
    del blocks
    pairs = competition_columns(pairs)
    log("candidate pairs", len(pairs), " avg/S1", len(pairs) / len(gt))

    s1_meta = load_fields(a.work, "train", [1], None, None, cols=["entity_id", "country"])
    country_of = pd.Series(s1_meta.country.values, index=s1_meta.entity_id.values)
    rng = np.random.RandomState(42)
    keep_path = f"{a.work}/{a.blocks}_s1.npy"
    s1_keep, tag = None, ""
    if os.path.exists(keep_path):
        s1_keep = np.load(keep_path)
        pool_ids = s1_keep
        tag = "_" + a.blocks
        log(f"S1 subset regime: {len(s1_keep)} kept S1 (others' records act as distractors)")
    elif "_sub" in a.blocks:  # small experimental block files
        pool_ids = np.intersect1d(s1_meta.entity_id.values, pairs.s1.unique())
    else:
        pool_ids = s1_meta.entity_id.values
    perm = pool_ids[rng.permutation(len(pool_ids))]
    val_ids = perm[:a.n_val]
    tr_ids = perm[a.n_val:a.n_val + a.n_train]
    if a.train_country:
        tr_ids = tr_ids[country_of.reindex(tr_ids).values == a.train_country]
    if a.val_country:
        val_ids = val_ids[country_of.reindex(val_ids).values == a.val_country]

    # blocking recall on val
    vp = pairs[pairs.s1.isin(val_ids)]
    tot = sum(len(gt.get(s, ())) for s in val_ids)
    hit = sum(c in gt.get(s, ()) for s, c in zip(vp.s1.values, vp.cand.values))
    log(f"blocking recall (val) = {hit / tot:.4f}; avg candidates/S1 = {len(vp) / len(val_ids):.2f}")

    # learn the extra-token table on S1 entities disjoint from train/val seeds (no target leakage)
    tab_ids = perm[a.n_val + a.n_train: a.n_val + a.n_train + a.n_table]
    if len(tab_ids) < 1000:
        tab_ids = perm[a.n_val:a.n_val + a.n_train][: a.n_table]
    TP = pairs[pairs.s1.isin(tab_ids)]
    c1 = load_fields(a.work, "train", [1], set(TP.s1.unique()), cols=["entity_id", "core"])
    c2 = load_fields(a.work, "train", [2, 3], set(TP.cand.unique()), cols=["entity_id", "core"])
    m1 = pd.Series(c1.core.values, index=c1.entity_id.values)
    m2 = pd.Series(c2.core.values, index=c2.entity_id.values)
    ylab = np.array([c in gt.get(s, ()) for s, c in zip(TP.s1.values, TP.cand.values)])
    xtok = learn_extra_token_table(m1.reindex(TP.s1.values).values, m2.reindex(TP.cand.values).values, ylab)
    del c1, c2, m1, m2, TP
    log(f"extra-token table: {len(xtok)} tokens; e.g. " +
        ", ".join(f"{t}={xtok[t]:.2f}" for t in ["group", "exports", "ventures", "services", "center", "dba"] if t in xtok))
    pickle.dump(xtok, open(f"{a.work}/extra_tok.pkl", "wb"))

    pairs_c = pairs[["s1", "cand"]].copy()
    pairs_c["country"] = country_of.reindex(pairs_c.s1.values).values

    def closure(seed_ids):
        if a.no_closure:
            P = pairs[pairs.s1.isin(seed_ids)].copy()
            P["country"] = country_of.reindex(P.s1.values).values
            return P
        cands = pairs.cand[pairs.s1.isin(seed_ids)].unique()
        P = pairs[pairs.cand.isin(cands)].copy()
        P["country"] = country_of.reindex(P.s1.values).values
        return P

    data = {}
    for name, ids in [("train", tr_ids), ("val", val_ids)]:
        P = closure(ids)
        F = featurize_pairs(P, a.work, "train", a.workers, s1_keep, tag, agree_src=pairs_c)
        P["y"] = np.array([c in gt.get(s, ()) for s, c in zip(P.s1.values, P.cand.values)], dtype=np.int8)
        P["is_seed"] = P.s1.isin(ids).values
        data[name] = (P.reset_index(drop=True), F.reset_index(drop=True))
        log(f"{name}: pairs={len(P)} seed-rows={P.is_seed.sum()} pos-rate={P.y[P.is_seed].mean():.3f}")

    P, F = data["train"]
    drop = set(x for x in a.drop_feats.split(",") if x)
    cols = [c for c in F.columns if not (a.no_closure and c.endswith("_relc")) and c not in drop]
    m = P.is_seed.values
    Pv, Fv = data["val"]
    models = []
    for k in range(a.n_models):
        kind, model = make_model(a.iters, a.workers)
        if kind == "lgb":
            import lightgbm as lgb
            model.set_params(random_state=k, subsample=0.8 if k == 0 else 0.7,
                             colsample_bytree=0.8 if k == 0 else 0.6)
            mv = Pv.is_seed.values
            model.fit(F.loc[m, cols].values, P.y.values[m],
                      eval_set=[(Fv.loc[mv, cols].values, Pv.y.values[mv])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
            log(f"model {k}: best iteration", model.best_iteration_)
        else:
            model.set_params(random_state=k)
            model.fit(F.loc[m, cols].values, P.y.values[m])
        models.append(model)
    model = models[0] if len(models) == 1 else Ensemble(models)
    log("trained", kind, "rows", m.sum(), "features", len(cols))
    unseen = None
    if a.loco:
        unseen = train_unseen_country_model(P, F, Pv, Fv, cols, gt, a)
    if not a.save_train_feats:
        data["train"] = (data["train"][0], None)
        del F
        import gc; gc.collect()

    pv = model.predict_proba(Fv[cols].values)[:, 1]
    (bt, bf), rows = tune_threshold(Pv.s1.values, Pv.cand.values, pv, gt, list(val_ids))
    for t, f in rows:
        log(f"  thr={t:.3f}  F0.5={f:.5f}")
    log(f"BEST val macro F0.5 = {bf:.5f} at thr={bt}")
    f_no = macro_f05_int(postprocess(Pv.s1.values, Pv.cand.values, pv, bt, False), gt, list(val_ids))
    log(f"  (same thr without one-to-one: {f_no:.5f})")
    os.makedirs(f"{a.work}/models", exist_ok=True)
    pickle.dump({"model": model, "cols": cols, "thr": bt, "K": a.K, "side": a.side, "kind": kind, "iters": a.iters,
                 "unseen": unseen, "train_countries": sorted(set(Pv.country.unique()) | set(P.country.unique()))},
                open(f"{a.work}/models/gbdt_stage1.pkl", "wb"))
    for name in ("train", "val"):
        P, F = data[name]
        P.to_pickle(f"{a.work}/{name}_pairs.pkl")
        if name == "val" or a.save_train_feats:
            F.to_pickle(f"{a.work}/{name}_feats.pkl")
    np.save(f"{a.work}/val_pred_stage1.npy", pv)
    try:
        imp = getattr(models[0], "feature_importances_", None)
        if imp is not None:
            for c, v in sorted(zip(cols, imp), key=lambda x: -x[1])[:25]:
                log(f"   imp {c}: {v}")
    except Exception:
        pass
    log("done")


if __name__ == "__main__":
    main()


def decide_expected_f(s1, cand, prob, m=0.0, one_to_one=True, floor=0.05):
    """per-S1 subset selection maximising expected F0.5 given (roughly calibrated) probabilities.
    E[F_k] ~= 1.25*S_k / (0.25*(S_all+m) + k);  E[F_0] ~= prod(1-p)."""
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": prob})
    if one_to_one:
        # a candidate can only belong to its best S1
        best = d.groupby("cand").p.transform("max")
        d = d[(d.p >= best) & (d.p >= floor)]
    else:
        d = d[d.p >= floor]
    d = d.sort_values(["s1", "p"], ascending=[True, False])
    out = {}
    s_v, c_v, p_v = d.s1.values, d.cand.values, d.p.values
    starts = np.flatnonzero(np.r_[True, s_v[1:] != s_v[:-1]])
    ends = np.r_[starts[1:], len(s_v)]
    for a0, a1 in zip(starts, ends):
        p = p_v[a0:a1]
        S_all = p.sum() + m
        cs = np.cumsum(p)
        k = np.arange(1, len(p) + 1)
        ef = 1.25 * cs / (0.25 * S_all + k)
        e0 = np.prod(1 - p)
        j = int(np.argmax(ef))
        if ef[j] > e0:
            out[s_v[a0]] = list(c_v[a0:a0 + j + 1])
    return out
