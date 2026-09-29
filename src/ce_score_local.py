"""Score the v7 candidate pairs that the Kaggle cross-encoder run did not cover (local GPU).

The Kaggle job scored every v6 test pair (ce_test_0..7.parquet) and every v6 validation pair (ce_val.parquet).
v7 selects candidates differently, so part of its pairs are new.  This script finds them and scores them with
the same fine-tuned model (ce_model/), writing ce_test_new.parquet and ce_val_new.parquet next to the others.

python src/ce_score_local.py --work WORK --ce_dir KAGGLE_OUT [--blocks_test blocks_test_v7 --blocks_train blocks_train_v7]
"""
import argparse
import glob
import time

from common import *
from data import load_fields
from kaggle_ce import texts_for


def pair_hash(s, c):
    a = pd.util.hash_array(np.asarray(s, dtype=np.int64))
    b = pd.util.hash_array(np.asarray(c, dtype=np.int64) + np.int64(7919))
    return a ^ (b * np.uint64(0x9E3779B97F4A7C15))


def new_pairs(P, files):
    have = np.concatenate([pair_hash(d.s1.values, d.cand.values)
                           for d in (pd.read_parquet(f, columns=["s1", "cand"]) for f in files)])
    have.sort()
    h = pair_hash(P.s1.values, P.cand.values)
    pos = np.searchsorted(have, h)
    pos[pos >= len(have)] = 0
    return P[have[pos] != h].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ce_dir", required=True)
    ap.add_argument("--blocks_test", default="blocks_test_v7")
    ap.add_argument("--blocks_train", default="blocks_train_v7")
    ap.add_argument("--n_val", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--max_len", type=int, default=96)
    a = ap.parse_args()

    s1_meta = load_fields(a.work, "train", [1], None, None, cols=["entity_id", "country"])
    val_ids = s1_meta.entity_id.values[np.random.RandomState(42).permutation(len(s1_meta))][:a.n_val]
    del s1_meta
    Bv = load_df(f"{a.work}/{a.blocks_train}", ["s1", "cand"])
    Bv = Bv[Bv.s1.isin(val_ids)].reset_index(drop=True)
    Nv = new_pairs(Bv, [f"{a.ce_dir}/ce_val.parquet"])
    Bt = load_df(f"{a.work}/{a.blocks_test}", ["s1", "cand"])
    old_test = sorted(f for f in glob.glob(f"{a.ce_dir}/ce_test_*.parquet") if f[-9:-8].isdigit())
    Nt = new_pairs(Bt, old_test)
    log(f"val pairs {len(Bv)} -> new {len(Nv)};  test pairs {len(Bt)} -> new {len(Nt)}")
    del Bv, Bt

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev)
    tok = AutoTokenizer.from_pretrained(f"{a.ce_dir}/ce_model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{a.ce_dir}/ce_model").to(dev).eval()
    if dev == "cuda":
        model = model.half()

    def score(A, B, name):
        out = np.zeros(len(A), dtype=np.float32)
        order = np.argsort([len(x) + len(z) for x, z in zip(A, B)])
        t1 = time.time()
        with torch.no_grad():
            for k, i in enumerate(range(0, len(A), a.bs)):
                ix = order[i:i + a.bs]
                enc = tok(list(A[ix]), list(B[ix]), truncation=True, max_length=a.max_len, padding=True,
                          return_tensors="pt").to(dev)
                out[ix] = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
                if k % 500 == 0:
                    log(f"  {name}: {i + len(ix)}/{len(A)} ({(i + len(ix)) / (time.time() - t1 + 1e-6):.0f} pairs/s)")
        return out

    if len(Nv):
        va, vb = texts_for(a.work, "train", Nv.s1.values, Nv.cand.values)
        Nv["ce"] = score(va, vb, "val")
        Nv.to_parquet(f"{a.ce_dir}/ce_val_new.parquet", index=False)
        log("val new pairs scored")
    step = 2_000_000
    outs = []
    for st in range(0, len(Nt), step):
        C = Nt.iloc[st:st + step].reset_index(drop=True)
        A, B = texts_for(a.work, "test", C.s1.values, C.cand.values)
        C["ce"] = score(A, B, f"test{st // step}")
        outs.append(C)
    if outs:
        pd.concat(outs, ignore_index=True).to_parquet(f"{a.ce_dir}/ce_test_new.parquet", index=False)
    log("ALL DONE")


if __name__ == "__main__":
    main()
