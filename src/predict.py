"""Step 4: score test candidate pairs, post-process, write output/matching_results.tsv and
output/candidate_pairs.tsv.

python src/predict.py --work work --data dataset --out output
"""
import argparse
import os
import pickle

from common import *
from data import load_fields, int_to_id
from features import base_features, context_features, competition_columns
from train import cut_candidates, get_idfs, postprocess, build_agree_table

CTX_BASE = ["cos_c3_nm", "cos_c3_ad", "name_x_addr", "cos_w_core"]


def score_split(split, work, workers, bundle, chunk_s1=120000, blocks_name=None):
    blocks = load_df(f"{work}/{blocks_name or 'blocks_' + split}")
    pairs = cut_candidates(blocks, bundle["K"], bundle["side"])
    del blocks
    pairs = competition_columns(pairs)
    s1_meta = load_fields(work, split, [1], None, None, cols=["entity_id", "country"])
    country_of = pd.Series(s1_meta.country.values, index=s1_meta.entity_id.values)
    pairs["country"] = country_of.reindex(pairs.s1.values).values
    log(f"{split}: candidate pairs {len(pairs)}  avg/S1 {len(pairs) / len(s1_meta):.2f}")
    model, cols = bundle["model"], bundle["cols"]
    probs = np.zeros(len(pairs), dtype=np.float32)
    tmp = f"{work}/tmp_feats"
    os.makedirs(tmp, exist_ok=True)
    for country, Pc in pairs.groupby("country"):
        build_agree_table(work, split, country, Pc)
        idfs = get_idfs(work, split, country, workers)
        s1u = Pc.s1.unique()
        chunks = [s1u[i:i + chunk_s1] for i in range(0, len(s1u), chunk_s1)]
        base_small = []
        files = []
        for ci, ch in enumerate(chunks):
            Pch = Pc[Pc.s1.isin(ch)]
            s1f = load_fields(work, split, [1], set(ch), country)
            cf = load_fields(work, split, [2, 3], set(Pch.cand.unique()), country)
            log(f"  {split}/{country} chunk {ci + 1}/{len(chunks)}: {len(Pch)} pairs")
            F = base_features(Pch, s1f, cf, workers, idfs)
            F.index = Pch.index
            fp = f"{tmp}/{split}_{country}_{ci}.pkl"
            F.to_pickle(fp)
            files.append(fp)
            base_small.append(F[CTX_BASE])
            del F, s1f, cf
        small = pd.concat(base_small).loc[Pc.index]
        ctx = context_features(Pc.reset_index(drop=True), small.reset_index(drop=True).copy())
        ctx.index = Pc.index
        ctx = ctx.drop(columns=CTX_BASE)
        for fp in files:
            F = pd.read_pickle(fp)
            F = F.join(ctx.loc[F.index])
            mdl = model
            if bundle.get("unseen") and country not in bundle.get("train_countries", []):
                mdl = bundle["unseen"]["model"]
            probs[pairs.index.get_indexer(F.index)] = mdl.predict_proba(F[cols].values)[:, 1]
            os.remove(fp)
            del F
    pairs["p"] = probs
    return pairs, s1_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--model", default="models/gbdt_stage1.pkl")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr_unseen", type=float, default=0.0, help="0=use LOCO estimate, -1=disable")
    ap.add_argument("--blocks", default=None)
    a = ap.parse_args()
    bundle = pickle.load(open(f"{a.work}/{a.model}", "rb"))
    thr = a.thr if a.thr is not None else bundle["thr"]
    pairs, s1_meta = score_split(a.split, a.work, a.workers, bundle, blocks_name=a.blocks)
    pairs[["s1", "cand", "p", "country"]].to_pickle(f"{a.work}/{a.split}_scored_stage1.pkl")
    p_adj = pairs.p.values.astype(np.float32).copy()
    if bundle.get("unseen") and a.thr_unseen >= 0:
        tu = a.thr_unseen if a.thr_unseen > 0 else bundle["unseen"]["thr"]
        un = ~np.isin(pairs.country.values, bundle.get("train_countries", []))
        # rescale so that a single global cut `thr` applies: p' = p * thr / tu on unseen countries
        p_adj[un] = np.minimum(p_adj[un] * (thr / tu), 0.999)
        log(f"unseen countries: {sorted(set(pairs.country.values[un]))} thr={tu}")
    matches = postprocess(pairs.s1.values, pairs.cand.values, p_adj, thr)
    s1_all = s1_meta.entity_id.values
    s1_str = int_to_id(s1_all)
    cand_map = pairs.groupby("s1").cand.apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)
    conv = lambda lst: list(int_to_id(lst)) if len(lst) else []
    with open(f"{a.out}/matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s, ss in zip(s1_all, s1_str):
            f.write(ss + "\t" + ",".join(conv(matches.get(s, []))) + "\n")
    with open(f"{a.out}/candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s, ss in zip(s1_all, s1_str):
            f.write(ss + "\t" + ",".join(conv(cand_map.get(s, []))) + "\n")
    n_m = sum(len(v) for v in matches.values())
    log(f"wrote outputs: {len(s1_all)} S1 rows, {n_m} matches, {len(pairs)} candidate pairs, thr={thr}")


if __name__ == "__main__":
    main()
